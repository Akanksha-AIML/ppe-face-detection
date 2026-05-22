#change 20may 18:22
"""
reid_tracker.py
---------------
Re-Identification module to assign stable person IDs across frames,
even when a person leaves and re-enters the frame.

Stack:
  - ByteTrack  : handled by ultralytics (in views.py via model.track())
  - OSNet      : feature extractor for appearance embedding (torchreid)
  - Cosine similarity gallery : matches returning persons to known IDs

Install dependencies:
    pip install torchreid
    # OSNet weights are auto-downloaded by torchreid on first run, OR
    # manually place osnet_x0_25_imagenet.pth in ./model/

How it connects to views.py:
    from .reid_tracker import ReidTracker
    reid_tracker = ReidTracker()                      # global, once
    stable_id = reid_tracker.get_stable_id(track_id, crop)   # per detection
"""

import numpy as np
import cv2
import torch
import threading
from collections import OrderedDict
import time

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("[ReidTracker] WARNING: torchreid not installed. "
          "Falling back to colour-histogram features. "
          "Install with: pip install torchreid")



# Constants — tune these for your scene

# COSINE_THRESHOLD   = 0.75  # similarity >= this → same person  (lower = stricter)
# MAX_GALLERY_SIZE   = 200    # max unique stable IDs kept in memory
# GALLERY_EXPIRY_SEC = 120    # seconds before an unseen person is forgotten
# TRACK_EXPIRY_SEC = 30
# MIN_CROP_SIZE      = 32     # ignore crops smaller than this (noisy detections)
# PROBATION_FRAMES   = 6      # frames to accumulate before first gallery match
# EMA_UPDATE_THRESH  = 0.82   # only update gallery embedding on confident matches
COSINE_THRESHOLD        = 0.75   # ongoing EMA update threshold (known person)
GALLERY_MATCH_THRESHOLD = 0.82   # first-commit threshold — stricter, avoids mislabeling
RECLAIM_THRESHOLD       = 0.87   # bypass active-exclusion if same person, very confident
MAX_GALLERY_SIZE        = 200
GALLERY_EXPIRY_SEC      = 120
TRACK_EXPIRY_SEC        = 30
MIN_CROP_SIZE           = 32
PROBATION_FRAMES        = 15     # was 6 — model runs every 3rd frame, need ~5 unique crops
EMA_UPDATE_THRESH       = 0.82


class _ColourHistFeatureExtractor:
    """
    Fallback extractor when torchreid is not available.
    Uses a normalised HSV colour histogram as the appearance descriptor.
    Good enough for simple scenes, not robust to lighting changes.
    """
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_bgr, (64, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)


class _OsNetFeatureExtractor:
    """
    OSNet-based appearance feature extractor via torchreid.
    Produces a 512-d L2-normalised embedding per crop.
    """
    def __init__(self, weights_path: str = ""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # build model
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",        # lightweight variant — good speed/accuracy tradeoff
            num_classes=1000,          # ImageNet pretraining
            pretrained=True,           # auto-downloads weights if not cached
        )

        # load custom weights if provided
        if weights_path:
            try:
                torchreid.utils.load_pretrained_weights(self.model, weights_path)
                print(f"[ReidTracker] Loaded OSNet weights from {weights_path}")
            except Exception as e:
                print(f"[ReidTracker] Could not load custom weights: {e}. Using ImageNet pretrained.")

        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocessing constants (ImageNet mean/std)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop_bgr, (128, 256))          # OSNet default input size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)                    # HWC → CHW
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(crop_bgr)
        feat   = self.model(tensor)                     # (1, 512)
        feat   = feat.cpu().numpy().flatten()
        norm   = np.linalg.norm(feat)
        return feat / (norm + 1e-6)                     # L2 normalise


 
# Gallery entry

class _GalleryEntry:
    def __init__(self, stable_id: int, embedding: np.ndarray):
        self.stable_id  = stable_id
        self.embedding  = embedding     # running average embedding
        self.last_seen  = time.time()
        self.seen_count = 1

    def update(self, new_embedding: np.ndarray, alpha: float = 0.9):
        """Exponential moving average of embeddings for robustness."""
        self.embedding  = alpha * self.embedding + (1 - alpha) * new_embedding
        norm            = np.linalg.norm(self.embedding)
        self.embedding  = self.embedding / (norm + 1e-6)
        self.last_seen  = time.time()
        self.seen_count += 1



# Main class — use this in views.py

class ReidTracker:
    """
    Wraps ByteTrack IDs (short-lived, per-session) with stable Re-ID.

    Usage in views.py:
        reid_tracker = ReidTracker()                         # once, globally
        stable_id = reid_tracker.get_stable_id(track_id, crop_bgr)
    """

    def __init__(self, weights_path: str = ""):
        # feature extractor
        if TORCHREID_AVAILABLE:
            self._extractor = _OsNetFeatureExtractor(weights_path)
            print("[ReidTracker] Using OSNet (torchreid) for Re-ID.")
        else:
            self._extractor = _ColourHistFeatureExtractor()
            print("[ReidTracker] Using colour-histogram fallback for Re-ID.")

        # gallery: stable_id → _GalleryEntry
        self._gallery: OrderedDict[int, _GalleryEntry] = OrderedDict()

        # maps current ByteTrack track_id → stable_id (cleared per session)
        # self._track_to_stable: dict[int, int] = {}
        self._track_to_stable: dict[int, tuple[int, float]] = {}
        self._probation: dict[int, list] = {}   # track_id → list of embeddings

        self._next_stable_id = 1
        self._lock = threading.Lock()

        # background thread to expire old gallery entries
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

    # ── public API ───────────────────────────────────────────────────────────

    # def get_stable_id(self, track_id: int, crop_bgr: np.ndarray) -> int:
    #     """
    #     Given a ByteTrack track_id and the person crop from the current frame,
    #     returns a stable Re-ID that persists even if the person leaves and
    #     re-enters (getting a new track_id from ByteTrack).

    #     Args:
    #         track_id  : ByteTrack ID from result.boxes.id
    #         crop_bgr  : person bounding-box crop (numpy BGR image)

    #     Returns:
    #         stable_id : integer, consistent across re-appearances
    #     """
    #     # reject tiny / broken crops
    #     if crop_bgr is None or crop_bgr.size == 0:
    #         return track_id
    #     h, w = crop_bgr.shape[:2]
    #     if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
    #         return track_id

    #     embedding = self._extractor.extract(crop_bgr)

    #     with self._lock:
    #         # 1. if we already mapped this ByteTrack ID, update gallery & return
    #         if track_id in self._track_to_stable:
    #             # stable_id = self._track_to_stable[track_id]
    #             stable_id, _ = self._track_to_stable[track_id]

    #             # update last seen time
    #             self._track_to_stable[track_id] = (stable_id, time.time())
    #             if stable_id in self._gallery:
    #                 self._gallery[stable_id].update(embedding)
    #             return stable_id

    #         # 2. new ByteTrack ID — try to match against existing gallery
    #         best_stable_id, best_sim = self._find_best_match(embedding)

    #         if best_sim >= COSINE_THRESHOLD:
    #             # recognised returning person
    #             stable_id = best_stable_id
    #             self._gallery[stable_id].update(embedding)
    #             print(f"[ReidTracker] track_id {track_id} → returning person "
    #                   f"stable_id {stable_id} (sim={best_sim:.3f})")
    #         else:
    #             # genuinely new person
    #             stable_id = self._next_stable_id
    #             self._next_stable_id += 1
    #             self._gallery[stable_id] = _GalleryEntry(stable_id, embedding)
    #             print(f"[ReidTracker] track_id {track_id} → NEW stable_id {stable_id}")

    #         # self._track_to_stable[track_id] = stable_id
    #         self._track_to_stable[track_id] = (stable_id, time.time())

    #         # keep gallery bounded
    #         self._evict_if_needed()

    #     return stable_id
    # ── REPLACE get_stable_id entirely ───────────────────────────────────────────

    def get_stable_id(self, track_id: int, crop_bgr: np.ndarray) -> int:
        """
        #     Given a ByteTrack track_id and the person crop from the current frame,
        #     returns a stable Re-ID that persists even if the person leaves and
        #     re-enters (getting a new track_id from ByteTrack).
    
        #     Args:
        #         track_id  : ByteTrack ID from result.boxes.id
        #         crop_bgr  : person bounding-box crop (numpy BGR image)
    
        #     Returns:
        #         stable_id : integer, consistent across re-appearances
        #     """

        if track_id is None:
           return -1      
        

        if crop_bgr is None or crop_bgr.size == 0:
            return -1
        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return -1
    
        embedding = self._extractor.extract(crop_bgr)
    
        with self._lock:
            # ── A. Already have a stable mapping for this track ───────────────
            if track_id in self._track_to_stable:
                stable_id, _ = self._track_to_stable[track_id]
                self._track_to_stable[track_id] = (stable_id, time.time())
    
                if stable_id in self._gallery:
                    entry = self._gallery[stable_id]
                    sim   = float(np.dot(embedding, entry.embedding))
                    # Only update gallery on confident observations
                    if sim >= EMA_UPDATE_THRESH:
                        entry.update(embedding)
                    else:
                        entry.last_seen = time.time()   # keep alive, don't drift
    
                return stable_id
    
            # ── B. New track_id — accumulate into probation buffer ────────────
            if track_id not in self._probation:
                self._probation[track_id] = []
    
            self._probation[track_id].append(embedding)
    
            if len(self._probation[track_id]) < PROBATION_FRAMES:
                # Not enough frames yet — return a temporary placeholder
                # (caller shows "Detecting..." during this phase)
                return -(track_id)          # negative = "not yet stable"
    
            # ── C. Probation complete — use averaged embedding for matching ────
            avg_embedding = np.mean(self._probation.pop(track_id), axis=0)
            norm          = np.linalg.norm(avg_embedding)
            avg_embedding = avg_embedding / (norm + 1e-6)
    
            best_stable_id, best_sim = self._find_best_match_exclusive(avg_embedding)
    
            if best_sim >= GALLERY_MATCH_THRESHOLD:
                stable_id = best_stable_id
                self._gallery[stable_id].update(avg_embedding)
                print(f"[ReidTracker] track_id {track_id} → returning person "
                      f"stable_id {stable_id} (sim={best_sim:.3f})")
            else:
                stable_id = self._next_stable_id
                self._next_stable_id += 1
                self._gallery[stable_id] = _GalleryEntry(stable_id, avg_embedding)
                print(f"[ReidTracker] track_id {track_id} → NEW stable_id {stable_id}")
    
            self._track_to_stable[track_id] = (stable_id, time.time())
            self._evict_if_needed()
    
            return stable_id

    def reset(self):
        """Call this if you restart a camera stream to clear ByteTrack mappings."""
        with self._lock:
            self._track_to_stable.clear()
            print("[ReidTracker] Track→stable mapping reset.")

    def clear_gallery(self):
        """Wipe the full Re-ID memory (new session, different location, etc.)."""
        with self._lock:
            self._gallery.clear()
            self._track_to_stable.clear()
            self._next_stable_id = 1
            print("[ReidTracker] Gallery cleared.")

    # ── private helpers ───────────────────────────────────────────────────────

    # def _find_best_match(self, embedding: np.ndarray):
    #     best_sim      = -1.0
    #     best_stable_id = -1
    #     for stable_id, entry in self._gallery.items():
    #         sim = float(np.dot(embedding, entry.embedding))   # cosine (both L2-normed)
    #         if sim > best_sim:
    #             best_sim      = sim
    #             best_stable_id = stable_id
    #     return best_stable_id, best_sim
    

    # ── ADD this method to ReidTracker (alongside _find_best_match) ───────────────

    def _find_best_match_exclusive(self, embedding: np.ndarray):
        """
        Like _find_best_match but skips gallery entries already owned
        by an active (non-stale) track_id. Prevents identity collapse.
        """
        # Build the set of stable_ids currently locked by live tracks
        now = time.time()
        active_stable_ids = {
            sid
            for _, (sid, t) in self._track_to_stable.items()
            if (now - t) < TRACK_EXPIRY_SEC
        }
    
        best_sim       = -1.0
        best_stable_id = -1
    
        # for stable_id, entry in self._gallery.items():
        #     if stable_id in active_stable_ids:
        #         continue        # owned by a live track — skip
    
        #     sim = float(np.dot(embedding, entry.embedding))
        #     if sim > best_sim:
        #         best_sim       = sim
        
        #         best_stable_id = stable_id
        for stable_id, entry in self._gallery.items():
            sim = float(np.dot(embedding, entry.embedding))
        
            if stable_id in active_stable_ids:
                # Same person got a new track_id from ByteTrack (e.g. after brief occlusion).
                # Allow reclaiming their own stable_id only if match is very high confidence.
                if sim >= RECLAIM_THRESHOLD and sim > best_sim:
                    best_sim       = sim
                    best_stable_id = stable_id
                continue       # skip at normal threshold — owned by a different live track
        
            if sim > best_sim:
                best_sim       = sim
                best_stable_id = stable_id
        return best_stable_id, best_sim



    def _evict_if_needed(self):
        """Remove oldest entries if gallery exceeds MAX_GALLERY_SIZE."""
        while len(self._gallery) > MAX_GALLERY_SIZE:
            oldest_id = next(iter(self._gallery))
            del self._gallery[oldest_id]
            print(f"[ReidTracker] Evicted stable_id {oldest_id} (gallery full).")

    def _cleanup_loop(self):
        """Background thread: expire gallery entries not seen for GALLERY_EXPIRY_SEC."""
        while True:
            time.sleep(30)
            now = time.time()
            with self._lock:
                self._cleanup_tracks()
                expired = [
                    sid for sid, entry in self._gallery.items()
                    if (now - entry.last_seen) > GALLERY_EXPIRY_SEC
                ]
                for sid in expired:
                    del self._gallery[sid]
                    print(f"[ReidTracker] Expired stable_id {sid} "
                          f"(not seen for {GALLERY_EXPIRY_SEC}s).")
                self._cleanup_tracks()    



    def _cleanup_tracks(self):
            """Remove stale track_id mappings."""
            now = time.time()
            expired = [
                tid for tid, (_, t) in self._track_to_stable.items()
                if (now - t) > TRACK_EXPIRY_SEC
            ]
            for tid in expired:
                del self._track_to_stable[tid]
                print(f"[ReidTracker] Removed stale track_id {tid}")        
