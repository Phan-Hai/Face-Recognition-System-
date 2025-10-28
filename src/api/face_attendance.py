import cv2
import numpy as np
import threading
import time
import copy
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing.image_processor import ImagePreprocessor
from src.face_detection.face_detector import FaceDetector
from src.database.milvus_client import MilvusClient
from configs.database import milvus_config


class FaceAttendance:
    def __init__(self, cam_index: int = 0, process_every_n: int = 2):
        """Refactored FaceAttendance with separate capture and processing threads.

        Args:
            cam_index: Index of the camera to open (default 0).
            process_every_n: Only run backend recognition once every N frames to reduce load.
        """
        self.image_processor = ImagePreprocessor()
        self.face_detector = FaceDetector()
        self.db_client = MilvusClient(
            host=milvus_config["host"],
            port=milvus_config["port"]
        )

        # Camera and threading controls
        self.cam_index = cam_index
        self.cap = None
        self.latest_frame = None
        self.latest_frame_index = 0
        self.frame_lock = threading.Lock()

        # Overlay results (list of dicts with recognition info and bbox)
        self.overlay = []
        self.overlay_lock = threading.Lock()

        self.stop_event = threading.Event()

        # How often to process (in frames)
        self.process_every_n = max(1, int(process_every_n))

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process one snapshot frame and return recognized faces with bbox list.

        Returns list of dicts: {'employee_id','name','department','position','confidence','bbox'}
        """
        variants = self.image_processor.generate_variants(frame)

        all_embeddings = []
        all_bboxes = []

        # Try original first
        faces_aligned, embs, bboxes = self.face_detector.detect_and_align(frame)
        if faces_aligned and embs:
            all_embeddings.extend(embs)
            all_bboxes.extend(bboxes)

        # If none, try variants
        if not all_embeddings:
            for variant in variants:
                faces_aligned, embs, bboxes = self.face_detector.detect_and_align(variant['image'])
                if faces_aligned and embs:
                    all_embeddings.extend(embs)
                    all_bboxes.extend(bboxes)
                    break

        recognized_faces = []
        if all_embeddings:
            for embedding, bbox in zip(all_embeddings, all_bboxes):
                matches = self.db_client.search_face(
                    collection_name=milvus_config["collection_name"],
                    query_embedding=embedding,
                    top_k=1
                )

                if matches and matches[0]['distance'] < 0.8:
                    match = matches[0]
                    recognized_faces.append({
                        'employee_id': match['employee_id'],
                        'name': match['name'],
                        'department': match['department'],
                        'position': match['position'],
                        'confidence': 1 - match['distance'],
                        'bbox': bbox
                    })

        return recognized_faces

    def _draw_overlay(self, frame: np.ndarray, overlay: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and display recognized names at fixed position."""
        info_blocks = []

        # --- Duyệt qua từng khuôn mặt nhận diện ---
        for info in overlay:
            bbox = info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                # Vẽ khung đỏ quanh khuôn mặt
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Thu thập thông tin hiển thị
            name = info.get('name', 'Unknown')
            emp_id = info.get('employee_id', '')
            department = info.get('department', '')
            position = info.get('position', '')
            conf = info.get('confidence', 0.0) * 100

            info_lines = [
                f"Name: {name} - ID: {emp_id}",
                f"Department: {department}",
                f"Position: {position}",
                f"Confidence: {conf:.1f}%"
            ]
            info_blocks.append(info_lines)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        try:
            font = ImageFont.truetype("arialbd.ttf", 14) 
        except:
            font = ImageFont.load_default()

        y_start = 30
        for block in info_blocks:
            for line in block:
                draw.text((10, y_start), line, font=font, fill=(255, 0, 0))
                y_start += 25
            y_start += 10

        # Convert lại sang OpenCV
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame


    def _capture_loop(self):
        """Continuously read frames from camera and store the latest frame and index."""
        try:
            self.cap = cv2.VideoCapture(self.cam_index)
            if not self.cap.isOpened():
                print("Không thể mở camera!")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                with self.frame_lock:
                    # store a copy to avoid modifications
                    self.latest_frame = frame.copy()
                    self.latest_frame_index += 1

                # Small sleep to yield
                time.sleep(0.005)
        finally:
            # release camera here if capture loop exits unexpectedly
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()

    def _processing_loop(self):
        """Process a snapshot of the latest frame every N frames and update overlay."""
        last_processed_index = 0
        while not self.stop_event.is_set():
            # grab a snapshot of latest frame and index
            with self.frame_lock:
                frame_idx = self.latest_frame_index
                frame_snapshot = None if self.latest_frame is None else self.latest_frame.copy()

            if frame_snapshot is None:
                time.sleep(0.01)
                continue

            if frame_idx - last_processed_index >= self.process_every_n:
                # run recognition on snapshot
                results = self.process_frame(frame_snapshot)
                with self.overlay_lock:
                    self.overlay = results
                last_processed_index = frame_idx
            else:
                # sleep a bit until next check
                time.sleep(0.005)

    def run(self):
        """Start camera, threads and display loop."""
        print("\nĐang khởi động camera (non-blocking capture) ...")
        print("Nhấn 'q' để thoát")

        # start capture and processing threads
        capture_t = threading.Thread(target=self._capture_loop, daemon=True)
        processing_t = threading.Thread(target=self._processing_loop, daemon=True)
        capture_t.start()
        processing_t.start()

        try:
            while not self.stop_event.is_set():
                with self.frame_lock:
                    display_frame = None if self.latest_frame is None else self.latest_frame.copy()

                if display_frame is None:
                    # no frame yet
                    time.sleep(0.02)
                    continue

                # draw overlay (use latest overlay snapshot)
                with self.overlay_lock:
                    overlay_snapshot = copy.deepcopy(self.overlay)

                frame_to_show = self._draw_overlay(display_frame, overlay_snapshot)

                cv2.imshow('Face Attendance System', frame_to_show)

                # If there are recognized faces we can also print a simple log (but only when overlay changes)
                if overlay_snapshot:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] Recognized {len(overlay_snapshot)} face(s)")
                    for face in overlay_snapshot:
                        print(f" - {face.get('name')} | ID: {face.get('employee_id')} | Dept: {face.get('department')} | Conf: {face.get('confidence',0.0)*100:.2f}%")

                # handle key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

        finally:
            # cleanup
            self.stop_event.set()
            # threads are daemon so they'll exit with process; join briefly if needed
            time.sleep(0.1)
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            try:
                self.db_client.disconnect()
            except Exception:
                pass