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
        # 학습한 고해상도 OBB 모델 경로 확인 필요
        self.model_yolo = YOLO("runs/obb/train/weights/best.pt")
        self.reader = easyocr.Reader(['en'], gpu=True)
        
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

    def order_points(self, pts):
        """OBB 4개 점을 [좌상, 우상, 우하, 좌하] 순서로 정렬하여 반환"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # x+y 합이 가장 작은 것이 좌상단, 가장 큰 것이 우하단
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # y-x 차이가 가장 작은 것이 우상단, 가장 큰 것이 좌하단
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

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
                # YOLO 추론 (imgsz=800으로 설정하여 정밀도 유지)
                results = self.model_yolo(img_input, verbose=False, conf=0.1, imgsz=800) 
                
                found = False
                for result in results:
                    if result.obb is None or len(result.obb) == 0:
                        continue

                    for obb in result.obb:
                        # 1. OBB 4개 꼭짓점 좌표 추출 및 정렬
                        raw_points = obb.xyxyxyxy.cpu().numpy()[0].astype(np.float32)
                        ordered_points = self.order_points(raw_points)
                        
                        # 2. 거리 계산을 위한 높이(H) 추출 (기존 로직 유지)
                        x1, y1 = int(np.min(ordered_points[:, 0])), int(np.min(ordered_points[:, 1]))
                        x2, y2 = int(np.max(ordered_points[:, 0])), int(np.max(ordered_points[:, 1]))
                        
                        h_pixel = y2 - y1
                        dist = (REF_DISTANCE_CM * REF_HEIGHT_PX) / h_pixel if h_pixel > 0 else 0

                        self.detection_result["box"] = (x1, y1, x2, y2)
                        self.detection_result["dist_cm"] = dist
                        self.detection_result["detected"] = True
                        found = True
                        
                        # 3. 비스듬한 번호판 정면으로 펴기 (Warp Perspective)
                        if dist < 120.0:
                            try:
                                # 목적지 이미지 크기 (번호판 비율 4:1에 맞춤)
                                width, height = 400, 100
                                dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                                
                                # 변환 행렬 계산 및 이미지 워핑
                                matrix = cv2.getPerspectiveTransform(ordered_points, dst_pts)
                                warped_img = cv2.warpPerspective(img_input, matrix, (width, height))
                                
                                # 가독성을 높이기 위한 전처리 (그레이스케일)
                                warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                                
                                # OCR 수행 (펴진 이미지를 사용함)
                                ocr_texts = self.reader.readtext(warped_gray, detail=0)
                                if len(ocr_texts) > 0:
                                    # 공백 제거 및 인식 결과 저장
                                    self.detection_result["text"] = "".join(ocr_texts).replace(" ", "")
                                    log_print(f"인식된 번호판: {self.detection_result['text']}")
                            except Exception as e:
                                log_print(f"OCR 워핑 에러: {e}")
                        
                        break 
                    if found: break

                if not found:
                    self.detection_result["detected"] = False
                    self.detection_result["box"] = None
                    self.detection_result["text"] = ""
            
            except Exception as e:
                log_print(f"🔥 감지 스레드 에러: {e}")
                time.sleep(1.0)
                
            time.sleep(0.01)