import threading
import time
import cv2
import numpy as np
import os
from datetime import datetime
from SCSCtrl import TTLServo


class OCRTask(threading.Thread):
    def __init__(self, robot, camera, ocr_detector, servo, widgets_dict, map1, map2):
        super().__init__()

        self.robot = robot
        self.camera = camera
        self.ocr_detector = ocr_detector
        self.servo = servo
        self.widgets_dict = widgets_dict
        self.map1 = map1
        self.map2 = map2
        
        self.trigger = False  # 외부에서 이 플래그를 True로 바꾸면 OCR 수행
        self.is_running = True
        self.result_text = "대기 중"
        
    def rotate_camera(self, angle, servo_id):
        if self.servo:
            self.servo.servoAngleCtrl(servo_id, angle, 1, 100) 
            time.sleep(2) # 이동 대기

    def process_ocr(self):

        # 1. 카메라를 인식 위치로 회전 (4번, 5번 서보 사용)
        print("[OCR] 카메라 인식 위치로 회전 중...")
        self.rotate_camera(60, 4)
        self.rotate_camera(45, 5)
        
        # 서보 이동 후 화면이 안정될 때까지 아주 짧은 대기 (선택 사항)
        time.sleep(0.5)

        # 2. 이미지 획득
        frame = self.camera.value
        result_text = "No Text Found" # 기본값 설정

        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if frame is None:
            print("[OCR] 이미지 획득 실패")
            result_text = "Capture Error"
        else:
            # 3. OCR 수행 (에러가 나도 카메라는 복구해야 하므로 try-except 사용)
            try:
                print("[OCR] Clova API 분석 시작...")
                result = self.ocr_detector.detect(frame, target='187고1604')
                
                if result and hasattr(result, 'text'):
                    print(f"[OCR] 인식 성공: {result.text}")
                    result_text = result.text
            except Exception as e:
                print(f"[OCR] 에러 발생: {e}")
                result_text = "API Error"

        # 4. 카메라 원위치 복귀 (핵심: return 이전에 실행되어야 함)
        print("[OCR] 작업 완료, 카메라 정면 복귀")
        self.rotate_camera(0, 4)
        self.rotate_camera(20, 5)
        
        # 5. 최종 결과 반환
        return result_text

    def run(self):
        global road_following_active

        while self.is_running:
            if self.trigger:
                print("[OCR] 작업 시작...")
                
                # 1. 주행 잠시 멈춤
                was_driving = road_following_active
                road_following_active = False
                self.robot.stop()
                time.sleep(0.5)

                # 2. 핵심 함수 호출 (회전 + 촬영 + OCR)
                # 이제 take_snapshot_and_ocr 내부에서 모든 비전 처리가 일어납니다.
                self.result_text = self.take_snapshot_and_ocr()
                
                print(self.result_text)

                
                self.trigger = False # 트리거 리셋
                if was_driving:
                    road_following_active = True # 주행 상태 복구
                    
            time.sleep(0.1)