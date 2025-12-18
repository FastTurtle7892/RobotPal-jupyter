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
        self.steering_model.load_state_dict(torch.load('best_steering_model_xy_test_12_17.pth', map_location=self.device))
        self.steering_model = self.steering_model.to(self.device).eval().half()
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

        # --- YOLO & OCR 모델 로드 ---
        log_print(">>> YOLO 및 OCR 모델 로딩 중...")
        self.model_yolo = YOLO("runs/obb/train/weights/best.pt")
        self.reader = easyocr.Reader(['ko', 'en'], gpu=True)
        
        # --- 스레드 공유 변수 ---
        self.stop_thread = False
        self.detection_thread = None
        self.latest_image_lock = threading.Lock()
        self.shared_latest_image = None
        
        # OCR 제어 플래그
        self.request_ocr = False
        self.ocr_pending = False
        
        self.detection_result = {
            "box": None,
            "dist_cm": None,
            "detected": False,
            "text": "" 
        }
        log_print(">>> 비전 시스템 초기화 완료")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def update_image(self, image):
        with self.latest_image_lock:
            self.shared_latest_image = image

    def start_detection_thread(self):
        if self.detection_thread is not None:
            self.stop_thread = True
            self.detection_thread.join()
        self.stop_thread = False
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.start()

    def stop_detection_thread(self):
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
                results = self.model_yolo(img_input, verbose=False, conf=0.1, imgsz=800) 
                found = False
                for result in results:
                    if result.obb is None or len(result.obb) == 0: continue

                    for obb in result.obb:
                        raw_points = obb.xyxyxyxy.cpu().numpy()[0].astype(np.float32)
                        ordered_points = self.order_points(raw_points)
                        
                        x1, y1 = int(np.min(ordered_points[:, 0])), int(np.min(ordered_points[:, 1]))
                        x2, y2 = int(np.max(ordered_points[:, 0])), int(np.max(ordered_points[:, 1]))
                        
                        h_pixel = y2 - y1
                        dist = (REF_DISTANCE_CM * REF_HEIGHT_PX) / h_pixel if h_pixel > 0 else 0

                        self.detection_result["box"] = (x1, y1, x2, y2)
                        self.detection_result["dist_cm"] = dist
                        self.detection_result["detected"] = True
                        found = True
                        
                        # 요청이 있을 때만 딱 한 번 OCR 수행
                        if self.request_ocr and not self.ocr_pending:
                            self.ocr_pending = True
                            try:
                                width, height = 400, 100
                                dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                                matrix = cv2.getPerspectiveTransform(ordered_points, dst_pts)
                                warped_img = cv2.warpPerspective(img_input, matrix, (width, height))
                                warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                                
                                ocr_texts = self.reader.readtext(warped_gray, detail=0)
                                if len(ocr_texts) > 0:
                                    self.detection_result["text"] = "".join(ocr_texts).replace(" ", "")
                                self.request_ocr = False 
                            except Exception as e:
                                log_print(f"OCR 에러: {e}")
                            finally:
                                self.ocr_pending = False
                        break 
                    if found: break

                if not found:
                    self.detection_result["detected"] = False
                    self.detection_result["box"] = None
            
            except Exception as e:
                time.sleep(1.0)
            time.sleep(0.01)