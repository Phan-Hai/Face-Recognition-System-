from src.api.face_attendance import FaceAttendance

def main():
    # Khởi tạo hệ thống chấm công
    attendance_system = FaceAttendance()
    
    print("=== Hệ thống chấm công khuôn mặt ===")
    print("Đang khởi động...")
    
    # Chạy hệ thống
    try:
        attendance_system.run()
    except KeyboardInterrupt:
        print("\nĐã dừng hệ thống!")
    except Exception as e:
        print(f"\nLỗi: {str(e)}")

if __name__ == "__main__":
    main()