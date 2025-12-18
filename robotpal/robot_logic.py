# robot_logic.py
import time
import cv2
import numpy as np
from robotpal import bgr8_to_jpeg
from . import robot_config as cfg
from .robot_vision import VisionSystem

class DrivingController:
    def __init__(self, vision_system: VisionSystem):
        self.vision = vision_system
        self.angle = 0.0
        self.angle_last = 0.0
        self.stop_end_time = 0.0
        self.is_stopped = False
        self.ignore_detection_until = 0.0
        
        # 제어 파라미터
        self.base_speed = 0.32
    
    def execute(self, change):
        try:
            image = change['new']
            current_time = time.time()
            
            # 비전 시스템에 이미지 업데이트 (스레드용)
            self.vision.update_image(image)
            
            # 시각화용 이미지 복사
            vis_image = image.copy()
            
            # 감지 결과 가져오기
            det_res = self.vision.detection_result
            cur_box = det_res["box"]
            cur_dist = det_res["dist_cm"]
            is_detected = det_res["detected"]
            cur_text = det_res["text"]

            # --- [상태 1] 정지 모드 ---
            if self.is_stopped:
                if current_time < self.stop_end_time:
                    cv2.putText(vis_image, "STOP MODE", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    remain_time = self.stop_end_time - current_time
                    cv2.putText(vis_image, f"{remain_time:.1f}s", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    if cur_box is not None:
                        x1, y1, x2, y2 = cur_box
                        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    
                    cfg.robot.left_motor.value = 0.0
                    cfg.robot.right_motor.value = 0.0
                    cfg.image_widget.value = bgr8_to_jpeg(vis_image)
                    return
                else:
                    self.is_stopped = False
                    self.ignore_detection_until = current_time + 3.0 
                    cfg.log_print(">>> 주행 재개 (1.5초간 쿨다운)")

            # --- [상태 2] 주행 중 감지 ---
            if is_detected and cur_dist is not None:
                x1, y1, x2, y2 = cur_box
                color = (0, 255, 255) if current_time < self.ignore_detection_until else (0, 255, 0)
                
                cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 정지 조건 (150cm 미만 & 쿨다운 지남)
                if cur_dist < 150.0 and current_time > self.ignore_detection_until:
                    self.is_stopped = True
                    self.stop_end_time = current_time + 1.5
                    
                    cfg.log_print(f"🛑 정지! (거리: {cur_dist:.1f}cm) / 번호판: [{cur_text}]")
                        
                    cfg.robot.left_motor.value = 0.0
                    cfg.robot.right_motor.value = 0.0
                    cfg.image_widget.value = bgr8_to_jpeg(vis_image)
                    return

            cfg.image_widget.value = bgr8_to_jpeg(vis_image)

            # --- [상태 3] 자율 주행 (ResNet) ---
            # VisionSystem의 preprocess 사용
            xy = self.vision.steering_model(self.vision.preprocess(image)).detach().float().cpu().numpy().flatten()
            
            x = xy[0]
            y = (0.5 - xy[1]) / 2.0
            
            cfg.speed_slider.value = self.base_speed
            
            self.angle = np.arctan2(x, y)
            pid = self.angle * 0.2 + (self.angle - self.angle_last) * 0.5
            self.angle_last = self.angle
            
            cfg.steering_slider.value = pid + 0
            left_val = max(min(cfg.speed_slider.value + cfg.steering_slider.value, 1.0), -0.9)
            right_val = max(min(cfg.speed_slider.value - cfg.steering_slider.value, 1.0), -0.9)
            
            cfg.robot.left_motor.value = float(left_val)
            cfg.robot.right_motor.value = float(right_val)
            
        except Exception as e:
            cfg.robot.left_motor.value = 0.0
            cfg.robot.right_motor.value = 0.0
            cfg.log_print(f"🚨 에러 발생: {e}")
            time.sleep(1.0)