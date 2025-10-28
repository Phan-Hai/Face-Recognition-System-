import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self, model_name: str = "buffalo_l"):
        """
        Khởi tạo FaceDetector với InsightFace
        Args:
            model_name: Tên model (buffalo_l là model lớn cho độ chính xác cao)
        """
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)  # 0 for first GPU, -1 for CPU
    
    def detect_and_align(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Phát hiện và căn chỉnh khuôn mặt từ ảnh
        Args:
            image: Ảnh đầu vào (BGR)
        Returns:
            Tuple của (danh sách khuôn mặt đã căn chỉnh, danh sách embedding vectors, danh sách bounding boxes (x,y,w,h))
        """
        # Nếu ảnh là grayscale (1 channel), chuyển về BGR để phù hợp với insightface
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image_for_detect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_for_detect = image

        # Phát hiện khuôn mặt
        faces = self.app.get(image_for_detect)

        if not faces:
            return [], [], []

        aligned_faces = []
        embeddings = []
        bboxes = []

        for face in faces:
            # Lấy landmarks
            landmarks = face.landmark_2d_106

            # Thử lấy bbox trực tiếp từ đối tượng face nếu có (tọa độ trên ảnh gốc)
            bbox = None
            if hasattr(face, 'bbox') and face.bbox is not None:
                try:
                    bx = int(face.bbox[0])
                    by = int(face.bbox[1])
                    bw = int(face.bbox[2] - face.bbox[0])
                    bh = int(face.bbox[3] - face.bbox[1])
                    bbox = (bx, by, bw, bh)
                except Exception:
                    bbox = None

            # Nếu không có bbox, tính từ landmarks
            if bbox is None and landmarks is not None:
                x, y, w, h = cv2.boundingRect(landmarks.astype(np.float32))
                bbox = (int(x), int(y), int(w), int(h))

            # Cắt và căn chỉnh khuôn mặt (sử dụng ảnh màu đã chuyển nếu cần)
            aligned_face = self._align_face(image_for_detect, landmarks)
            if aligned_face is not None:
                aligned_faces.append(aligned_face)
                embeddings.append(face.embedding)
                bboxes.append(bbox)
        
        return aligned_faces, embeddings, bboxes
    
    def _align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                    desired_size: int = 112) -> Optional[np.ndarray]:
        """
        Căn chỉnh khuôn mặt dựa trên landmarks
        Args:
            image: Ảnh gốc
            landmarks: 106 điểm landmarks
            desired_size: Kích thước mong muốn cho ảnh output
        Returns:
            Ảnh khuôn mặt đã được căn chỉnh
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        # Lấy các điểm mắt và miệng trung bình
        left_eye = landmarks[38:42].mean(axis=0)
        right_eye = landmarks[42:46].mean(axis=0)
        
        # Tính góc để căn chỉnh mắt ngang
        angle = -np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                    right_eye[0] - left_eye[0]))
        
        # Tính tâm của khuôn mặt
        center = landmarks.mean(axis=0)
        
        # Tạo ma trận transformation
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)
        
        # Xoay ảnh
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Cắt khuôn mặt
        bbox = cv2.boundingRect(landmarks.astype(np.float32))
        x, y, w, h = bbox
        face = rotated[int(y):int(y+h), int(x):int(x+w)]
        
        # Resize về kích thước mong muốn
        if face.size > 0:
            return cv2.resize(face, (desired_size, desired_size))
        
        return None