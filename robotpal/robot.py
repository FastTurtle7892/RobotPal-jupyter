# robotpal/robot.py

import traitlets
from traitlets.config.configurable import SingletonConfigurable
from .motor import Motor
from ._core.server import SimulatorServer


class Robot(SingletonConfigurable):
    """
    JetBot 호환 Robot 클래스
    """
    # [핵심] 사용자가 robot.left_motor.value 로 접근 가능하게 함
    left_motor = traitlets.Instance(Motor)
    right_motor = traitlets.Instance(Motor)

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)

        # 드라이버로 SimulatorServer 사용
        self.server = SimulatorServer.instance()

        # 모터 초기화 (Channel 1: Left, Channel 2: Right)
        self.left_motor = Motor(self.server, channel=1)
        self.right_motor = Motor(self.server, channel=2)

    def set_motors(self, left_speed, right_speed):
        self.left_motor.value = left_speed
        self.right_motor.value = right_speed

    def forward(self, speed=1.0, duration=None):
        self.left_motor.value = speed
        self.right_motor.value = speed

    def backward(self, speed=1.0):
        self.left_motor.value = -speed
        self.right_motor.value = -speed

    def left(self, speed=1.0):
        self.left_motor.value = -speed
        self.right_motor.value = speed

    def right(self, speed=1.0):
        self.left_motor.value = speed
        self.right_motor.value = -speed

    def stop(self):
        self.left_motor.value = 0
        self.right_motor.value = 0