import os
from src.api.face_registration import FaceRegistration

def main():
    # Khởi tạo đối tượng đăng ký khuôn mặt
    face_reg = FaceRegistration()
    
    # Đường dẫn đến thư mục chứa ảnh test
    test_image_dir = "data/test_images"
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
    
    # Thông tin nhân viên test
    employee_id = "HD_000004"
    name = "Phan Thị Mỹ Dung"
    department = "Nhân sự"
    position = "Truong phòng"
    
    # Đường dẫn ảnh test
    image_path = os.path.join(test_image_dir, "test_face_4.jpg")
    
    # Kiểm tra xem ảnh có tồn tại không
    if not os.path.exists(image_path):
        print(f"Vui lòng đặt ảnh test vào thư mục {test_image_dir} với tên 'test_face.jpg'")
        return
    
    # Thực hiện đăng ký
    num_faces, success_count = face_reg.register_face(image_path, employee_id, name=name, department=department, position=position)
    
    if num_faces == 0:
        print("Không tìm thấy khuôn mặt trong ảnh!")
    else:
        print(f"\nKết quả cuối cùng:")
        print(f"- Số khuôn mặt phát hiện được: {num_faces}")
        print(f"- Số khuôn mặt đăng ký thành công: {success_count}")
        
        if success_count > 0:
            print("\nKiểm tra thư mục 'aligned_faces' để xem các khuôn mặt đã được căn chỉnh")

if __name__ == "__main__":
    main()