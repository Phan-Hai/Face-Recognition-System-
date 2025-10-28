import cv2
import numpy as np
from typing import List, Dict
import random

class ImagePreprocessor:
    def __init__(self):
        # CLAHE với tham số chuẩn
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Kernel làm sắc nét hiệu quả nhất
        self.sharpen_kernels = [
            np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),  # Sharpen chuẩn
        ]
        
        # Các tham số Gaussian blur (giảm xuống 2 cấu hình hiệu quả nhất)
        self.gaussian_params = [
            {'ksize': (3, 3), 'sigma': 0},  # Làm mờ nhẹ
            {'ksize': (5, 5), 'sigma': 1},  # Làm mờ vừa
        ]
        
        # Các tham số độ sáng và độ tương phản (giảm xuống 2 cấu hình)
        self.brightness_contrast = [
            {'alpha': 1.2, 'beta': 10},   # Tăng độ sáng
            {'alpha': 0.8, 'beta': -10},  # Giảm độ sáng
        ]

    def generate_variants(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Tạo nhiều biến thể của ảnh với các phương pháp xử lý khác nhau
        Args:
            image: Ảnh đầu vào (BGR)
        Returns:
            List các biến thể của ảnh kèm thông tin xử lý
        """
        variants = []
        
        # Chuyển sang ảnh grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Biến thể gốc (convert về BGR để tương thích với face detector)
        variants.append({
            'image': cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR),
            'processing': 'original'
        })
        
        # 2. Các biến thể với Gaussian blur khác nhau
        for i, params in enumerate(self.gaussian_params):
            blurred = cv2.GaussianBlur(gray, params['ksize'], params['sigma'])
            variants.append({
                'image': cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
                'processing': f'gaussian_blur_k{params["ksize"][0]}_s{params["sigma"]}'
            })
            
            # Áp dụng CLAHE cho biến thể Gaussian đầu tiên
            if i == 0:
                enhanced = self.clahe.apply(blurred)
                variants.append({
                    'image': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                    'processing': f'gaussian_clahe_k{params["ksize"][0]}_s{params["sigma"]}'
                })
        
        # 3. Các biến thể với độ sáng và độ tương phản khác nhau
        for i, params in enumerate(self.brightness_contrast):
            adjusted = cv2.convertScaleAbs(gray, alpha=params['alpha'], beta=params['beta'])
            variants.append({
                'image': cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR),
                'processing': f'brightness_contrast_a{params["alpha"]}_b{params["beta"]}'
            })
            
            # Kết hợp với Gaussian blur
            gaussian_param = random.choice(self.gaussian_params)
            blurred_adjusted = cv2.GaussianBlur(adjusted, 
                                              gaussian_param['ksize'], 
                                              gaussian_param['sigma'])
            variants.append({
                'image': cv2.cvtColor(blurred_adjusted, cv2.COLOR_GRAY2BGR),
                'processing': f'bright_contrast_gaussian_a{params["alpha"]}_b{params["beta"]}'
            })
        
        # 4. Các biến thể với làm sắc nét khác nhau
        for i, kernel in enumerate(self.sharpen_kernels):
            sharpened = cv2.filter2D(gray, -1, kernel)
            variants.append({
                'image': cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
                'processing': f'sharpened_kernel{i+1}'
            })
            
            # Áp dụng CLAHE cho ảnh đã làm sắc nét
            enhanced_sharp = self.clahe.apply(sharpened)
            variants.append({
                'image': cv2.cvtColor(enhanced_sharp, cv2.COLOR_GRAY2BGR),
                'processing': f'sharpened_clahe'
            })
        
        # 5. Thêm nhiễu Gaussian (chỉ 1 mức nhiễu + denoise)
        noise_var = 10  # Mức nhiễu vừa phải
        noisy = gray.copy()
        noise = np.random.normal(0, noise_var, gray.shape).astype(np.uint8)
        noisy = cv2.add(noisy, noise)
        variants.append({
            'image': cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR),
            'processing': f'added_noise_var{noise_var}'
        })
        
        # Kết hợp với Gaussian blur để giảm nhiễu
        denoised = cv2.GaussianBlur(noisy, (5, 5), 1)
        variants.append({
            'image': cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
            'processing': 'noise_gaussian_denoised'
        })
        
        return variants