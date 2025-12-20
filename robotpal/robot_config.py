# # robot_config.py
# import ipywidgets
# import traitlets
# import time
# from robotpal import Robot, Camera, bgr8_to_jpeg
# from robotpal.SCSCtrl import TTLServo

# # --- 하드웨어 설정 ---
# robot = Robot()
# camera = Camera()

# # 거리 계산을 위한 기준 상수 (사용자 환경에 맞게 조정 가능)
# REF_DISTANCE_CM = 60.0
# REF_HEIGHT_PX = 110.0

# # --- UI 위젯 설정 ---
# # 조향 및 속도 모니터링을 위한 슬라이더
# speed_slider = ipywidgets.FloatSlider(value=0.0, min=-1.0, max=1.0, description='speed', orientation='vertical')
# steering_slider = ipywidgets.FloatSlider(value=0.0, min=-1.0, max=1.0, description='steering')
# image_widget = ipywidgets.Image(format='jpeg', width=500, height=500)
# log_widget = ipywidgets.Textarea(value='', layout=ipywidgets.Layout(width='100%', height='200px'))

# traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

# def log_print(message):
#     """Jupyter Textarea 위젯과 콘솔에 동시에 로그를 출력합니다."""
#     timestamp = time.strftime("[%H:%M:%S] ")
#     new_log = timestamp + str(message) + "\n"
#     log_widget.value = new_log + log_widget.value
#     print(new_log.strip())

# # --- 서보 제어 관련 추가 기능 ---

# def init_servos():
#     """시스템 시작 시 서보 모터의 초기 각도를 설정합니다."""
#     try:
#         # 예: 카메라 정면 (ID 1: 좌우, ID 5: 상하)
#         # 각도 값은 실제 로봇의 조립 상태에 따라 다를 수 있습니다.
#         TTLServo.servoAngleCtrl(1, 0, 1, 150)  # 좌우 정면
#         TTLServo.servoAngleCtrl(5, -25, 1, 150) # 상하 적정 높이
        
#         log_print(">>> 서보 모터 초기화 완료")
#     except Exception as e:
#         log_print(f"🚨 서보 초기화 에러: {e}")

# def set_servo(id, angle, speed=150):

#     time.sleep(1)
#     if(angle == 20.0):
#         TTLServo.servoAngleCtrl(id, angle, 1, speed)
#     elif(angle == 0.0):
#         TTLServo.servoAngleCtrl(id, angle, -1, speed)
#     time.sleep(1)

# # --- 유틸리티 ---
# def bgr8_to_jpeg(value, quality=75):
#     """OpenCV 이미지를 위젯용 JPEG 포맷으로 변환합니다."""
#     import cv2
#     return bytes(cv2.imencode('.jpg', value)[1])