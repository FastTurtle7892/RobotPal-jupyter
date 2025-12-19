# robot_logic.py
import time
import cv2
import numpy as np
import threading
from robotpal import bgr8_to_jpeg
from . import robot_config as cfg
from .robot_vision import VisionSystem

class DrivingController:
    def __init__(self, vision_system: VisionSystem):
        self.vision = vision_system
        self.angle = 0.0
        self.angle_last = 0.0
        
        self.state = "DRIVING"
        self.base_speed = 0.32
        self.ignore_until = 0.0
    
    def execute(self, change):
        try:
            image = change['new']
            current_time = time.time()
            
            # 1. 비전 시스템에 최신 이미지 전달 (백그라운드 감지용)
            self.vision.update_image(image)
            
            # 2. [핵심 수정] 시퀀스 수행 중(카메라 회전 등)일 때 처리
            if self.state == "SEQUENCE":
                # 카메라가 돌아가 있는 동안 모델이 조향을 계산하지 못하도록 즉시 차단
                cfg.robot.left_motor.value = 0.0
                cfg.robot.right_motor.value = 0.0
                return

            # 3. 감지 결과 확인
            det_res = self.vision.detection_result
            is_detected = det_res["detected"]
            cur_dist = det_res["dist_cm"]

            # 4. 번호판 감지 시 시퀀스 전환
            if is_detected and cur_dist is not None:
                if cur_dist < 120.0 and current_time > self.ignore_until:
                    self.state = "SEQUENCE"
                    # 시퀀스 스레드 시작 전, 메인 스레드에서 즉시 정지 명령 전송 (반응성 강화)
                    cfg.robot.left_motor.value = 0.0
                    cfg.robot.right_motor.value = 0.0
                    
                    threading.Thread(target=self.plate_recognition_sequence).start()
                    return

            # 5. --- 일반 주행 (ResNet) ---
            # state가 "DRIVING"이고 카메라가 정면일 때만 실행
            self.drive_step(image)
            
        except Exception as e:
            cfg.robot.stop()
            cfg.log_print(f"🚨 에러 발생: {e}")

    def plate_recognition_sequence(self):
        """정지 -> 카메라 회전 -> OCR -> 카메라 복귀 시퀀스"""
        try:
            cfg.log_print("🛑 번호판 인식! 정지합니다.")
            cfg.robot.stop()
            
            # 1. 카메라 오른쪽으로 20만큼 돌리기 (서보 ID 1 사용 가정)
            cfg.log_print("카메라 회전 중 (오른쪽 20)")
            cfg.set_servo(1, 20.0)
            time.sleep(0.8) 
            
            # 2. OCR 요청 및 대기
            cfg.log_print("OCR 수행 중...")
            self.vision.request_ocr = True
            
            # 결과가 나올 때까지 최대 3초 대기
            timeout = time.time() + 3.0
            while self.vision.request_ocr and time.time() < timeout:
                time.sleep(0.1)
            
            # 3. 로그창에 OCR 값 출력
            ocr_val = self.vision.detection_result["text"]
            cfg.log_print(f"OCR 인식 결과: [{ocr_val}]")
            
            # 4. 카메라 다시 왼쪽으로 20 돌려 복귀
            cfg.log_print("카메라 복귀 중...")
            cfg.set_servo(1, 0.0) # 원위치
            time.sleep(0.8)
            
            # 5. 주행 재개 설정
            self.ignore_until = time.time() + 3.0 # 다시 인식되지 않도록 3초 쿨다운
            self.state = "DRIVING"
            cfg.log_print("▶️ 주행을 재개합니다.")
            
        except Exception as e:
            cfg.log_print(f"시퀀스 에러: {e}")
            self.state = "DRIVING"

    def drive_step(self, image):
        """기존 .pth 모델 주행 로직"""
        xy = self.vision.steering_model(self.vision.preprocess(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]
        y = (0.5 - xy[1]) / 2.0
        
        self.angle = np.arctan2(x, y)
        pid = self.angle * 0.2 + (self.angle - self.angle_last) * 0.5
        self.angle_last = self.angle
        
        steering_val = pid
        speed_val = self.base_speed
        
        left_val = max(min(speed_val + steering_val, 1.0), -0.9)
        right_val = max(min(speed_val - steering_val, 1.0), -0.9)
        
        cfg.robot.left_motor.value = float(left_val)
        cfg.robot.right_motor.value = float(right_val)
        
        # UI 업데이트용 슬라이더 값 갱신
        cfg.speed_slider.value = speed_val
        cfg.steering_slider.value = steering_val