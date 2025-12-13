import traitlets
import ipywidgets
from ..image import bgr8_to_jpeg  # 방금 만든 변환 함수 임포트


class CameraBase(traitlets.HasTraits):
    value = traitlets.Any()  # 여기에는 이제 'Numpy 배열'이 들어갑니다!

    @staticmethod
    def instance(*args, **kwargs):
        raise NotImplementedError

    def widget(self):
        if hasattr(self, '_widget'):
            return self._widget

        image = ipywidgets.Image(format='jpeg', width=300, height=300)

        # [핵심] value(Numpy) -> 변환(JPEG) -> image.value
        traitlets.dlink((self, 'value'), (image, 'value'), transform=bgr8_to_jpeg)

        self._widget = image
        return image