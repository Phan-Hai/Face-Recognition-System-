import os
import cv2
import numpy as np
from datetime import datetime
from src.preprocessing.image_processor import ImagePreprocessor
from src.face_detection.face_detector import FaceDetector
from src.database.milvus_client import MilvusClient
from configs.database import milvus_config

class FaceRegistration:
    def __init__(self):
        self.image_processor = ImagePreprocessor()
        self.face_detector = FaceDetector()
        self.db_client = MilvusClient(
            host=milvus_config["host"],
            port=milvus_config["port"]
        )
        
        # Khởi tạo collection
        self.db_client.create_face_collection(
            collection_name=milvus_config["collection_name"],
            dim=milvus_config["vector_dim"]
        )
    
    def register_face(self, image_path: str, employee_id: str, 
                      name: str = None, department: str = None, position: str = None) -> tuple[int, int]:
        """
        Đăng ký khuôn mặt mới từ ảnh với nhiều biến thể
        Args:
            image_path: Đường dẫn đến ảnh
            employee_id: ID của nhân viên
        Returns:
            Tuple (số khuôn mặt phát hiện được, số khuôn mặt đăng ký thành công)
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh từ {image_path}")
            return (0, 0)
        
        try:
            # Tạo các biến thể của ảnh
            image_variants = self.image_processor.generate_variants(image)
            print(f"Đã tạo {len(image_variants)} biến thể của ảnh")
            
            total_faces = 0
            total_success = 0
            processed_embeddings = set()  # Để tránh trùng lặp embedding gần giống nhau
            
            # Xử lý từng biến thể
            for variant_idx, variant in enumerate(image_variants, 1):
                processed_image = variant['image']
                processing_info = variant['processing']
                
                # Phát hiện và căn chỉnh khuôn mặt
                # FaceDetector.detect_and_align returns (aligned_faces, embeddings, bboxes)
                aligned_faces, embeddings, bboxes = self.face_detector.detect_and_align(processed_image)
                
                if not aligned_faces or not embeddings:
                    continue
                
                num_faces = len(aligned_faces)
                total_faces += num_faces
                print(f"\nBiến thể {variant_idx} ({processing_info}):")
                print(f"Phát hiện được {num_faces} khuôn mặt")
                
                # Lưu các khuôn mặt vào database
                timestamp = int(datetime.now().timestamp())
                
                for face_idx, (face, embedding) in enumerate(zip(aligned_faces, embeddings)):
                    try:
                        # Kiểm tra xem embedding có quá giống với các embedding đã xử lý không
                        embedding_key = tuple(np.round(embedding, 3))
                        if embedding_key in processed_embeddings:
                            continue
                        processed_embeddings.add(embedding_key)
                        
                        # Tạo ID duy nhất cho mỗi biến thể của khuôn mặt
                        face_employee_id = f"{employee_id}_v{variant_idx}_f{face_idx+1}"
                        
                        # Lưu vào database (gửi kèm thông tin nhân viên)
                        self.db_client.insert_face(
                            collection_name=milvus_config["collection_name"],
                            employee_id=face_employee_id,
                            face_embedding=embedding,
                            timestamp=timestamp,
                            name=name,
                            department=department,
                            position=position
                        )
                        
                        # Lưu ảnh khuôn mặt đã căn chỉnh
                        variant_dir = os.path.join(os.path.dirname(image_path), 
                                                 "aligned_faces", 
                                                 processing_info)
                        os.makedirs(variant_dir, exist_ok=True)
                        face_path = os.path.join(variant_dir, f"{face_employee_id}.jpg")
                        cv2.imwrite(face_path, face)
                        
                        total_success += 1
                        print(f"Đăng ký thành công khuôn mặt {face_idx+1} trong biến thể {variant_idx}")
                        
                    except Exception as e:
                        print(f"Lỗi khi đăng ký khuôn mặt {face_idx+1} trong biến thể {variant_idx}: {str(e)}")
            
            print("\nKết quả tổng cộng:")
            print(f"- Số biến thể ảnh: {len(image_variants)}")
            print(f"- Tổng số khuôn mặt phát hiện: {total_faces}")
            print(f"- Số khuôn mặt unique đăng ký thành công: {total_success}")
            return (total_faces, total_success)
            
        except Exception as e:
            print(f"Lỗi khi đăng ký khuôn mặt: {str(e)}")
            # Trả về tuple mặc định để tránh lỗi unpack ở caller
            return (0, 0)
        
    def __del__(self):
        """
        Đóng kết nối database khi object bị hủy
        """
        self.db_client.disconnect()