#change 20may 
# face_detector.py
import pickle
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, embeddings_path: str, similarity_thresh: float = 0.45):
        self.thresh = similarity_thresh
        self.db = self._load_embeddings(embeddings_path)

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

    def _load_embeddings(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = pickle.load(f)

        normalised = {}
        for name, emb_list in data.items():
            emb = np.mean(np.array(emb_list), axis=0).astype(np.float32)
            normalised[name] = emb / (np.linalg.norm(emb) + 1e-6)

        print(f"[FaceDetector] Loaded {len(normalised)} identities")
        return normalised

    def identify_from_crop(self, crop) -> str:
        """
        Takes a cropped person image (numpy array), returns name string.
        Returns 'Unknown' if no face found or match below threshold.
        """
        if crop is None or crop.size == 0:
            return "Unknown"

        faces = self.app.get(crop)
        if not faces:
            return "Unknown"

        face = max(faces, key=lambda f: f.det_score)
        return self._identify(face.embedding)

    def _identify(self, embedding: np.ndarray) -> str:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        best_name, best_score = "Unknown", -1.0

        for name, ref in self.db.items():
            score = float(np.dot(embedding, ref))
            if score > best_score:
                best_score, best_name = score, name

        return best_name if best_score >= self.thresh else "Unknown"
# # face_detector.py
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis


# class FaceDetector:
#     def __init__(self, embeddings_path: str, similarity_thresh: float = 0.45):
#         self.thresh = similarity_thresh
#         self.db = self._load_embeddings(embeddings_path)

#         self.app = FaceAnalysis(
#             name="buffalo_l",
#             providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
#         )
#         self.app.prepare(ctx_id=0, det_size=(320, 320))

#     def _load_embeddings(self, path: str) -> dict:
#         with open(path, "rb") as f:
#             data = pickle.load(f)

#         normalised = {}
#         for name, emb_list in data.items():
#             emb = np.mean(np.array(emb_list), axis=0).astype(np.float32)
#             normalised[name] = emb / (np.linalg.norm(emb) + 1e-6)

#         print(f"[FaceDetector] Loaded {len(normalised)} identities")
#         return normalised

#     def identify_from_crop(self, crop) -> str:
#         """
#         Takes a cropped person image (numpy array), returns name string.
#         Returns 'Unknown' if no face found or match below threshold.
#         """
#         if crop is None or crop.size == 0:
#             return "Unknown"

#         faces = self.app.get(crop)
#         if not faces:
#             return "Unknown"

#         face = max(faces, key=lambda f: f.det_score)
#         return self._identify(face.embedding)

#     def _identify(self, embedding: np.ndarray) -> str:
#         embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
#         best_name, best_score = "Unknown", -1.0

#         for name, ref in self.db.items():
#             score = float(np.dot(embedding, ref))
#             if score > best_score:
#                 best_score, best_name = score, name

#         return best_name if best_score >= self.thresh else "Unknown"