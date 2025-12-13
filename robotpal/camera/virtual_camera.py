import traitlets
from traitlets.config.configurable import SingletonConfigurable
from .camera_base import CameraBase
from .._core.server import SimulatorServer


# [중요] SingletonConfigurable을 먼저 상속받아야 instance() 메서드가 정상 작동합니다.
class Camera(SingletonConfigurable, CameraBase):
    """
    RobotPal 시뮬레이터용 카메라 클래스
    """

    def __init__(self, *args, **kwargs):
        # MRO 순서에 따라 초기화
        super(Camera, self).__init__(*args, **kwargs)

        # 싱글톤 서버 연결
        self.server = SimulatorServer.instance()

        # 서버 이미지(BGR) -> 내 value(BGR) 연결
        traitlets.dlink((self.server, 'latest_image'), (self, 'value'))

    def start(self):
        pass

    def stop(self):
        pass