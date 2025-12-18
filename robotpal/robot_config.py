# robot_config.py
import ipywidgets
import ipywidgets.widgets as widgets
from datetime import datetime
from robotpal import Robot, Camera
from robotpal.SCSCtrl import TTLServo

# ==========================================
# 1. 시스템 상수 설정
# ==========================================
CAM_WIDTH = 816
CAM_HEIGHT = 616
REF_DISTANCE_CM = 60.0
REF_HEIGHT_PX = 110.0

# ==========================================
# 2. 로깅 위젯 및 함수
# ==========================================
# 로그를 표시할 Textarea 위젯
log_widget = ipywidgets.Textarea(
    value="",
    placeholder="로그가 여기에 표시됩니다...",
    description="📝 LOG:",
    disabled=True,
    layout=ipywidgets.Layout(width='600px', height='200px')
)

def log_print(msg):
    """텍스트 상자에 메시지를 직접 추가하는 함수"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    new_line = f"[{timestamp}] {msg}\n"
    log_widget.value = new_line + log_widget.value

# ==========================================
# 3. 하드웨어 초기화 (싱글톤처럼 사용)
# ==========================================
robot = Robot()
# 카메라는 main에서 인스턴스를 가져오거나 여기서 생성
camera = Camera.instance(width=CAM_WIDTH, height=CAM_HEIGHT)

# 서보 모터 초기화 함수
def init_servos():
    TTLServo.servoAngleCtrl(5, -25, 1, 100)
    TTLServo.servoAngleCtrl(1, 0, 1, 100)
    log_print(">>> 서보 모터 초기화 완료")

# ==========================================
# 4. 제어용 UI 위젯
# ==========================================
image_widget = ipywidgets.Image(format='jpeg', width=500, height=500)
steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering')
speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed')