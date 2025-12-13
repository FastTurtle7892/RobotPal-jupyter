# robotpal/motor.py

import traitlets
from traitlets.config.configurable import Configurable


class Motor(Configurable):
    """
    JetBot의 Motor 클래스 호환 구현체.
    값이 변경되면 SimulatorServer에 업데이트 요청을 보냄.
    """
    value = traitlets.Float()

    # JetBot 호환 설정값 (실제 동작엔 영향 X)
    alpha = traitlets.Float(default_value=1.0).tag(config=True)
    beta = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel, *args, **kwargs):
        super(Motor, self).__init__(*args, **kwargs)
        self._driver = driver  # SimulatorServer 인스턴스
        self._channel = channel  # 1(Left) or 2(Right)

    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """서버로 모터 값 전송"""
        # alpha/beta 적용 (JetBot 호환)
        mapped_value = value * self.alpha + self.beta
        self._driver.update_motor_value(self._channel, mapped_value)