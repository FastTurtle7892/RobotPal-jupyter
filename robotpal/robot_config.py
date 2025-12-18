# robot_config.py
import ipywidgets
import ipywidgets.widgets as widgets
from datetime import datetime
from robotpal import Robot, Camera
from robotpal.SCSCtrl import TTLServo
import time

# ==========================================
# 1. 시스템 상수 설정
# ==========================================
# 화질이 1632 x 1232로 변경됨에 따라 수정
CAM_WIDTH = 816
CAM_HEIGHT = 616
REF_DISTANCE_CM = 60.0
# 해상도가 2배 커졌으므로 기준 픽셀 높이도 2배인 220으로 수정
REF_HEIGHT_PX = 110.0 

# ==========================================
# 2. 로깅 위젯 및 함수 (원본 보존)
# ==========================================
log_widget = ipywidgets.Textarea(
    value="",
    placeholder="로그가 여기에 표시됩니다...",
    description="📝 LOG:",
    disabled=True,
    layout=ipywidgets.Layout(width='600px', height='200px')
)

def log_print(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    new_line = f"[{timestamp}] {msg}\n"
    log_widget.value = new_line + log_widget.value

# ==========================================
# 3. 하드웨어 초기화 및 추가 제어 함수
# ==========================================
robot = Robot()
camera = Camera.instance(width=CAM_WIDTH, height=CAM_HEIGHT)

def init_servos():
    # 5번: 상하 (기본 -25), 1번: 좌우 (기본 0)
    TTLServo.servoAngleCtrl(5, -25, 1, 100)
    TTLServo.servoAngleCtrl(1, 0, 1, 100)
    log_print(">>> 서보 모터 초기화 완료")

# [추가] 카메라를 오른쪽으로 20도 돌리는 함수 (ID 1번 서보가 좌우라고 가정)
def rotate_camera_for_ocr():
    log_print(">>> OCR 정밀 인식을 위해 카메라 20도 회전")
    # 현재 0도에서 오른쪽인 20도로 이동 (하드웨어 방향에 따라 -20일 수 있음)
    TTLServo.servoAngleCtrl(1, 20, 1, 150) 
    time.sleep(1.0) # 회전 후 안정화 대기

# ==========================================
# 4. 제어용 UI 위젯
# ==========================================
image_widget = ipywidgets.Image(format='jpeg', width=500, height=500)
steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering')
speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed')