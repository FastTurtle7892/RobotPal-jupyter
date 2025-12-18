# robot_vision.py
import threading
import time
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image
import easyocr
import numpy as np
from ultralytics import YOLO
from .robot_config import log_print, REF_DISTANCE_CM, REF_HEIGHT_PX

class VisionSystem:
    def __init__(self):
        self.device = torch.device('cuda')
        
        # --- ResNet (조향) 모델 로드 ---
        log_print(">>> 조향 모델 로딩 중...")
        self.steering_model = torchvision.models.resnet18(pretrained=False)
        self.steering_model.fc = torch.nn.Linear(512, 2)
        # 경로가 맞는지 확인하세요
        self.steering_model.load_state_dict(torch.load('best_steering_model_xy_test_12_17.pth', map_location=self.device))
        self.steering_model = self.steering_model.to(self.device).eval().half()
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

        # --- YOLO & OCR 모델 로드 ---
        log_print(">>> YOLO 및 OCR 모델 로딩 중...")
        self.model_yolo = YOLO("runs/obb/train/weights/best.pt")
        self.reader = easyocr.Reader(['en'], gpu=True) # OCR 리더 미리 로드 권장
        
        # --- 스레드 공유 변수 ---
        self.stop_thread = False
        self.detection_thread = None
        self.latest_image_lock = threading.Lock()
        self.shared_latest_image = None
        
        self.detection_result = {
            "box": None,
            "dist_cm": None,
            "detected": False,
            "text": "" 
        }
        log_print(">>> 비전 시스템 초기화 완료")

    def preprocess(self, image):
        """ResNet용 이미지 전처리"""
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def update_image(self, image):
        """메인 루프에서 최신 이미지를 받아옴"""
        with self.latest_image_lock:
            self.shared_latest_image = image

    def start_detection_thread(self):
        """감지 스레드 시작"""
        if self.detection_thread is not None:
            self.stop_thread = True
            self.detection_thread.join()
        
        self.stop_thread = False
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.start()

    def stop_detection_thread(self):
        """감지 스레드 중지"""
        self.stop_thread = True
        if self.detection_thread:
            self.detection_thread.join()

    def _detection_worker(self):
        log_print(">>> 감지 스레드 시작됨")
        while not self.stop_thread:
            img_input = None
            with self.latest_image_lock:
                if self.shared_latest_image is not None:
                    img_input = self.shared_latest_image.copy()
            
            if img_input is None:
                time.sleep(0.01)
                continue

            try:
                # YOLO 추론
                results = self.model_yolo(img_input, verbose=False, conf=0.1) 
                
                found = False
                for result in results:
                    # [수정 1] OBB 모델은 result.boxes 대신 result.obb를 사용합니다.
                    # 감지된 것이 없으면 result.obb는 None일 수 있습니다.
                    if result.obb is None:
                        continue

                    # result.obb를 반복합니다.
                    for obb in result.obb:
                        # [수정 2] OBB 좌표 변환
                        # OBB는 기울어진 사각형이므로 4개의 점(xyxyxyxy)을 줍니다.
                        # 기존 로직(거리 계산, OCR crop)을 위해 이를 포함하는 정방형 박스(x1,y1,x2,y2)로 변환합니다.
                        corners = obb.xyxyxyxy.cpu().numpy()[0] # shape: (4, 2)
                        
                        x_coords = corners[:, 0]
                        y_coords = corners[:, 1]
                        
                        x1 = int(np.min(x_coords))
                        y1 = int(np.min(y_coords))
                        x2 = int(np.max(x_coords))
                        y2 = int(np.max(y_coords))
                        
                        h_pixel = y2 - y1
                        dist = (REF_DISTANCE_CM * REF_HEIGHT_PX) / h_pixel if h_pixel > 0 else 0

                        self.detection_result["box"] = (x1, y1, x2, y2)
                        self.detection_result["dist_cm"] = dist
                        self.detection_result["detected"] = True
                        found = True
                        
                        # 120cm 이내 OCR 수행
                        if dist < 120.0:
                            try:
                                h, w, _ = img_input.shape
                                x1 = max(0, x1); y1 = max(0, y1)
                                x2 = min(w, x2); y2 = min(h, y2)
                                crop_img = img_input[y1:y2, x1:x2]
                                
                                if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                                    ocr_texts = self.reader.readtext(crop_img, detail=0)
                                    if len(ocr_texts) > 0:
                                        self.detection_result["text"] = " ".join(ocr_texts)
                            except: pass
                        
                        # 가장 가까운(또는 신뢰도 높은) 하나만 처리하고 break
                        break 
                    if found: break

                if not found:
                    self.detection_result["detected"] = False
                    self.detection_result["box"] = None
            
            except Exception as e:
                # 에러 로그가 너무 많이 뜨지 않게 1초 대기
                log_print(f"🔥 감지 스레드 에러: {e}")
                time.sleep(1.0)
                
            time.sleep(0.01)