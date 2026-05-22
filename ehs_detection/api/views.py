# change 20 may  18:21

from django.shortcuts import render
# Create your views here.
from .reid_tracker import ReidTracker
import cv2
from django.http import StreamingHttpResponse
from ultralytics import YOLO
import threading
import requests
from pytz import timezone
import time
import os
from datetime import datetime
import numpy as np
from .face_detector import FaceDetector
VIOLATION_COOLDOWN_SECONDS = 10

from concurrent.futures import ThreadPoolExecutor, Future

face_executor = ThreadPoolExecutor(max_workers=2)
face_futures: dict[int, Future] = {}          # stable_id → pending Future

# Separate cooldowns: Unknown persons re-checked every 2 s, known every 15 s
FACE_COOLDOWN_UNKNOWN = 2
FACE_COOLDOWN_KNOWN   = 15


trackid_to_name = {}
last_face_check = {}
FACE_COOLDOWN = 10

face_detector = FaceDetector(embeddings_path="../embeddings.pkl")


model = YOLO("./model/best_v26.pt")
reid_tracker = ReidTracker(weights_path="./model/osnet_x0_25.pt")

VIOLATION_SNAPSHOT_DIR = "Violation Snapshots"
os.makedirs(VIOLATION_SNAPSHOT_DIR, exist_ok=True)

# Step 7 (cooldown control — IMPORTANT)
last_saved = {}


CONFIDENCE_THRESHOLD = 0.6
API_URL = "http://127.0.0.1:8000/detection/image_upload/"
# LABELS = {
#     0: "Glasses", 1: "Gloves", 2: "Hardhat", 3: "Mask",
#     4: "No-Glasses", 5: "No-Gloves", 6: "No-Hardhat",
#     7: "No-Mask", 8: "No-Safety-Vest", 9: "No-Shoes",
#     10: "Safety Vest", 11: "Shoes", 12: "Other", 13: "Person"
# }
# ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

LABELS = {
    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
    4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
    7: "Safety Vest", 8: "machinery", 9: "vehicle"
}
COLORS = {0: (0, 255, 0), 1: (0, 255, 0),2:(0, 0, 255),3:(0, 0, 255),4:(0, 0, 255),5:(0, 255, 0),6:(0, 255, 0),7:(0, 255, 0),8:(0, 255, 0),9:(0, 255, 0)}  # 0: Empty, 1: Filled
PERSON_CLASS_ID = 5
first_saved_image = None

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import DetectionSerializer

class DetectionCreateAPI(APIView):
    def post(self, request):
        serializer = DetectionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Image saved successfully"},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Detection
from .serializers import DetectionListSerializer

class DetectionListAPI(APIView):
    def get(self, request):
        detections = Detection.objects.order_by("-created_at")[:3]
        serializer = DetectionListSerializer(
            detections,
            many=True,
            context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_200_OK)




def draw_boxes(frame, results):
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = LABELS.get(cls_id, "Unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(cls_id, (255, 255, 255))   # was always green — fixed
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)


def draw_person_tracks(frame, results, tracked_person_id=None):
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        print("Track IDs:", boxes.id)
        track_ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()

        for track_id, class_id, confidence, (x1, y1, x2, y2) in zip(
            track_ids, classes, confidences, coordinates
        ):
            if class_id != PERSON_CLASS_ID or confidence < CONFIDENCE_THRESHOLD:
                continue

            if tracked_person_id is not None and track_id != tracked_person_id:
                continue

            color = (0, 0, 255) if tracked_person_id == track_id else COLORS.get(class_id, (255, 255, 255))
            # label = f"Person ID {track_id} {confidence:.2f}"
            # Crop the person
            crop = frame[y1:y2, x1:x2]
            
            # Safety check (very important)
            if crop.size == 0:
                continue
            
            # Get stable ID from ReID
            stable_id = reid_tracker.get_stable_id(track_id, crop)
            if stable_id < 0:           # still in probation
                label_text = f"Detecting... | {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                continue                # skip to next box
            
            # Use stable ID instead of track_id
            label = f"Person ID {stable_id} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

def frame_generator():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        draw_boxes(frame, results)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def person_tracking_generator(source=1, tracked_person_id=None, weights_path="./model/git_model.pt"):
    tracker_model = load_model(weights_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = tracker_model.track(
                frame,
                persist=True,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml",
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU,
                verbose=False,
            )
            draw_person_tracks(frame, results, tracked_person_id=tracked_person_id)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()

#########################################
import cv2
import numpy as np
import os
import pathlib
import threading
import queue
import time
from ultralytics import YOLO
from datetime import datetime

# # Modify pathlib for compatibility-------------------this is for windows
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
import pathlib
import sys

# Fix for loading Windows-trained models on Linux ----------------------this is for linux
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath


# LABELS = {0: "Glasses",1:'Gloves', 2: "Hardhat",3:"Mask",4:"Safety Vest",5:"Shoes",6:"other",7:"person"}
# LABELS = {0: "Glasses",1:'Gloves', 2: "Hardhat",3:"Mask",4:"No-Glasses",5:"No-Gloves",6:"No-Hardhat",7:"No-Mask",8:"No-Safety-Vest",9:"No-Shoes",10:"Safety Vest",11:"Shoes",12:"other",13:"person"}

CONFIDENCE_THRESHOLD = 0.45 # Adjusted confidence threshold
NMS_IOU = 0.45

VIOLATION_CLASSES = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest
# Folder paths for detected and comparison images
output_folder = r'Detected Objects'
comparison_folder = r'Comparison Folder'
matched_folder = r'Complete Matches'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)
os.makedirs(matched_folder, exist_ok=True)

# Frame and processing queues for multiple cameras
frame_queues = {}
processed_queues = {}




def load_model(weights_path):
    """Load YOLOv8 model with exception handling."""
    try:
        print(f"Loading model from {weights_path}...")
        model = YOLO(weights_path)  # Load YOLOv8 model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_color_and_label(class_id):
    """Map class IDs to colors and labels."""
    color = COLORS.get(class_id, (255, 255, 255))  # Default color: White
    label = LABELS.get(class_id, f"Class {class_id}")  # Default label
    return color, label

def save_detected_object(frame, box, label, feed_id):
    """Save the detected object as an image in the output folder with a feed identifier."""
    x1, y1, x2, y2 = map(int, box[:4])
    cropped_image = frame[y1:y2, x1:x2]

    # Create a unique file name using timestamp and feed_id
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{feed_id}_{label}_{timestamp}.jpg"
    filepath = os.path.join(output_folder, filename)

    # Save the cropped image
    cv2.imwrite(filepath, cropped_image)
    print(f"Saved detected object from feed {feed_id}: {filepath}")

    # Check for matching images in the comparison folder
    match_images(filepath, timestamp)

def match_images(detected_path, timestamp):
    """Check for a matching image in the comparison folder based on timestamp and move to matched folder if found."""
    for filename in os.listdir(comparison_folder):
        if timestamp in filename:
            comparison_path = os.path.join(comparison_folder, filename)
            
            # Move both images to the matched folder
            matched_detected_path = os.path.join(matched_folder, f"Detected_{timestamp}.jpg")
            matched_comparison_path = os.path.join(matched_folder, f"Comparison_{timestamp}.jpg")
            
            os.rename(detected_path, matched_detected_path)
            os.rename(comparison_path, matched_comparison_path)
            print(f"Matched and moved: {matched_detected_path} and {matched_comparison_path}")

            # Display images side-by-side
            show_images_side_by_side(matched_detected_path, matched_comparison_path, timestamp)
            break

def show_images_side_by_side(img_path1, img_path2, timestamp):
    """Display two images side-by-side with their timestamp."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    img1 = cv2.resize(img1, (1080, 1920))
    img2 = cv2.resize(img2, (1080, 1920))

    # Concatenate images horizontally
    combined_img = np.hstack((img1, img2))

    # Add timestamp as text on the combined image
    cv2.putText(combined_img, timestamp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the combined image
    cv2.imshow('Complete Match', combined_img)
    cv2.waitKey(3000)  # Display for 3 seconds


def save_violation_snapshot(frame, label, person_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{person_name}_{label}_{timestamp}.jpg"
    filepath = os.path.join(VIOLATION_SNAPSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"[VIOLATION] Saved: {filepath}")


def draw_results(frame, results, feed_id):
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            if class_id not in LABELS or confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None

            color, label = get_color_and_label(class_id)

            #  PERSON LOGIC
           
            # if class_id == PERSON_CLASS_ID:
            #     if track_id is not None:
            #         crop = frame[y1:y2, x1:x2]

            #         if crop.size != 0:
            #             stable_id = reid_tracker.get_stable_id(track_id, crop)

            #             now = time.time()

            #             if stable_id not in trackid_to_name or \
            #                now - last_face_check.get(stable_id, 0) > FACE_COOLDOWN:

            #                 name = face_detector.identify_from_crop(crop)

            #                 trackid_to_name[stable_id] = name
            #                 last_face_check[stable_id] = now

            #             name = trackid_to_name.get(stable_id, "Detecting...")
            #             label_text = f"{name} | ID {stable_id} | {confidence:.2f}"
            #         else:
            #             label_text = f"Person ? {confidence:.2f}"
            #     else:
            #         label_text = f"Person ? {confidence:.2f}"
            if class_id == PERSON_CLASS_ID:
                if track_id is not None:
                    crop = frame[y1:y2, x1:x2]
            
                    if crop.size != 0:
                        stable_id = reid_tracker.get_stable_id(track_id, crop)

                        if stable_id < 0:
                            label_text = f"Detecting... | {confidence:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            continue   
            
                        now = time.time()
                        current_name = trackid_to_name.get(stable_id, "Detecting...")
            
                        # ── 1. Collect completed face futures ────────────────────────
                        if stable_id in face_futures and face_futures[stable_id].done():
                            try:
                                result = face_futures[stable_id].result()
                                trackid_to_name[stable_id] = result
                                last_face_check[stable_id] = now
                                current_name = result
                            except Exception:
                                pass
                            del face_futures[stable_id]
            
                        # ── 2. Submit new face task if cooldown passed & not pending ──
                        is_unknown  = (current_name in ("Detecting...", "Unknown"))
                        cooldown    = FACE_COOLDOWN_UNKNOWN if is_unknown else FACE_COOLDOWN_KNOWN
                        not_pending = stable_id not in face_futures
            
                        if not_pending and (now - last_face_check.get(stable_id, 0) > cooldown):
                            # Only use the upper 55 % of the crop — face lives there
                            face_crop = crop[: int(crop.shape[0] * 0.55), :]
                            if face_crop.size != 0:
                                face_futures[stable_id] = face_executor.submit(
                                    face_detector.identify_from_crop, face_crop.copy()
                                )
            
                        label_text = f"{current_name} | ID {stable_id} | {confidence:.2f}"
                    else:
                        label_text = f"Person ? {confidence:.2f}"
                else:
                    label_text = f"Person ? {confidence:.2f}"
            

         
            #  VIOLATION LOGIC
        
            else:
                label_text = f"{label} {confidence:.2f}"

                if class_id in VIOLATION_CLASSES:
                    now = time.time()

                    stable_id_or_track = track_id if track_id is not None else f"{x1}_{y1}"
                    key = f"{stable_id_or_track}_{class_id}"

                    if now - last_saved.get(key, 0) > VIOLATION_COOLDOWN_SECONDS:
                        last_saved[key] = now

                        person_name = trackid_to_name.get(stable_id_or_track, "Unknown")
                        save_violation_snapshot(frame.copy(), label, person_name)

         
            # DRAW (inside loop)
         
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )



def frame_reader(cap, feed_id):
    """Function to read frames from the RTSP stream in a separate thread for each feed."""
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1680, 920))  # Resize for consistency
            if not frame_queues[feed_id].full():
                frame_queues[feed_id].put(frame)
        time.sleep(0.03)  # Limit frame rate (e.g., 30 FPS)

def frame_processor(model, feed_id):
    """Function to process frames from the queue in a separate thread for each feed."""
    frame_count = 0
    last_results = None

    while True:
        if not frame_queues[feed_id].empty():
            frame = frame_queues[feed_id].get()
            frame_count += 1

            if frame_count % 3 == 0 or last_results is None:  # process every 3rd frame
                # last_results = model(frame)
                last_results = model.track(
                    frame,
                    persist=True,
                    tracker="ultralytics/cfg/trackers/bytetrack.yaml",
                    conf=CONFIDENCE_THRESHOLD,
                    iou=NMS_IOU,
                    verbose=False
                )

            if last_results:
                draw_results(frame, last_results, feed_id)  # reuse last results on skipped frames

            if not processed_queues[feed_id].full():
                processed_queues[feed_id].put(frame)

def start_feed_processing(feed_id, rtsp_url, model):
    """Initialize the feed processing for a single RTSP feed."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for feed {feed_id}.")
        return

    frame_queues[feed_id] = queue.Queue(maxsize=10)
    processed_queues[feed_id] = queue.Queue(maxsize=10)

    threading.Thread(target=frame_reader, args=(cap, feed_id), daemon=True).start()
    threading.Thread(target=frame_processor, args=(model, feed_id), daemon=True).start()

def save_detection(frame, label, timestamp,detected_id):
    """Save the detection image and send it to the API."""
    global first_saved_image

    # Add timestamp to the bottom of the frame
    formatted_timestamp = datetime.strptime(timestamp, '%Y%m%d-%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    text_position = (10, frame.shape[0] - 10)  # Bottom-left corner of the frame
    cv2.putText(frame, formatted_timestamp, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save the image
    formatted_label = label.replace(" ", "_")
    filename = f"{formatted_label}_{timestamp}.jpeg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved detection: {filepath}")

    # Check and display which image is saved first
    if first_saved_image is None:
        first_saved_image = label
        print(f"First saved image: {label}")

    # Start thread for API communication
    thread = threading.Thread(target=send_detection_to_api, args=(filepath, label, timestamp,detected_id))
    thread.start()

def send_detection_to_api(image_path, label, timestamp,detected_id):
    """Send detection data to the API asynchronously."""
    try:
        timestamp_obj = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
        timestamp_ist = timestamp_obj.astimezone(timezone("Asia/Kolkata"))
        data = {
            "assets": label,
            "is_safe" :1
        }
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(API_URL, data=data, files=files)
            if response.status_code == 201:
                print("Image successfully saved in database.")
            else:
                print(f"Failed to save image. Status code: {response.status_code}, Error: {response.json()}")
    except Exception as e:
        print(f"Error sending image to API: {e}")

def main(weights_path, rtsp_urls):
    """Main function to capture video from multiple RTSP feeds and display results."""
    model = load_model(weights_path)
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d-%H%M%S')
    # Start a processing thread for each RTSP feed
    for feed_id, rtsp_url in enumerate(rtsp_urls):
        print(f"Starting feed {feed_id} for URL: {rtsp_url}")
        start_feed_processing(feed_id, rtsp_url, model)

    while True:
        for feed_id in range(len(rtsp_urls)):
            if not processed_queues[feed_id].empty():
                frame = processed_queues[feed_id].get()
                # cv2.imshow(f'YOLOv8 Detection - Feed {feed_id}', frame)
                try:
                #   save_detection(frame,"SAFE",timestamp,11)
                    pass
                except:
                    pass
                ret, buffer = cv2.imencode('.jpg', frame)
                
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Check if any key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                return
                        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    weights_path = './model/best_v26.pt'  # Update with your YOLOv8 model path
    rtsp_urls = [1
        # 'rtsp://Dhiraj:Triazine$123@192.168.1.229/cam/realmonitor?channel=1&subtype=0'
        
        
    ]
    main(weights_path, rtsp_urls)
# 'rtsp://UserName:Password@IpAddress/cam/realmonitor?channel=4&subtype=0',
########################################

def video_feed(request):
    weights_path = './model/git_model.pt'  # Update with your YOLOv8 model path
    rtsp_urls = [1

#         # 'rtsp://admin:Tspl%25123@192.168.1.235:554/cam/realmonitor?channel=1&subtype=1'
        # 'rtsp://Dhiraj:Triazine$123@192.168.1.229/cam/realmonitor?channel=1&subtype=0'
# #         # 'rtsp://admin:Tspl%25123@192.168.1.235:554/cam/realmonitor?channel=1&subtype=1'
    ]
    return StreamingHttpResponse(
        # frame_generator(),
        main(weights_path, rtsp_urls),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def person_tracking_feed(request):
    person_id = request.GET.get("person_id")

    try:
        tracked_person_id = int(person_id) if person_id is not None else None
    except ValueError:
        return Response(
            {"error": "person_id must be an integer."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    return StreamingHttpResponse(
        person_tracking_generator(tracked_person_id=tracked_person_id),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
  
