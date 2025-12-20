import traitlets
from traitlets.config.configurable import SingletonConfigurable
from .camera_base import CameraBase
from .._core.server import SimulatorServer
import numpy as np
import cv2

class Camera(SingletonConfigurable, CameraBase):
    """
    [RobotPal Simulator Camera]
    JetBot OpenCvGstCamera와 동일한 인터페이스를 제공하며,
    설정된 해상도에 따라 자동으로 최적화된 데이터 스트림을 선택합니다.
    """
    
    # [JetBot Compatibility] 원본 코드의 변수명과 기본값을 그대로 유지
    width = traitlets.Integer(default_value=224).tag(config=True)
    height = traitlets.Integer(default_value=224).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    capture_width = traitlets.Integer(default_value=816).tag(config=True)
    capture_height = traitlets.Integer(default_value=616).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)
        
        # 싱글톤 서버 연결
        self.server = SimulatorServer.instance()
        
        # [Smart Stream Logic]
        # JetBot이 하드웨어 리사이징을 하듯, 요청 해상도에 따라 소스를 결정함
        
        if self.width == 224 and self.height == 224:
            # Case 1: 기본 AI 모드 (대부분의 경우)
            # 서버가 미리 224x224로 줄여놓은 데이터를 가져옴 (매우 빠름)
            traitlets.dlink((self.server, 'latest_image'), (self, 'value'))
        else:
            # Case 2: 고해상도 커스텀 모드 (예: width=640)
            # 원본 스트림을 가져와서 직접 디코딩 및 리사이즈 (느림, 화질 중심)
            traitlets.dlink((self.server, 'latest_jpeg'), (self, 'value'), transform=self._decode_custom_res)

        # [Display Stream]
        # 위젯 표시용 데이터는 무조건 '원본 JPEG'를 연결 (랙 없음, 고화질)
        traitlets.dlink((self.server, 'latest_jpeg'), (self, 'value_jpeg'))

    def _decode_custom_res(self, jpeg_bytes):
        """사용자 정의 해상도 처리를 위한 디코더"""
        if not jpeg_bytes: return None
        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 0)
                # 요청 크기와 다르면 리사이즈
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return frame
        except: pass
        return None

    # JetBot 호환용 더미 함수들
    def start(self): pass
    def stop(self): pass
    def restart(self): pass