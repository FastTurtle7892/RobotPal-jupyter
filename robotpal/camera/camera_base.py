import traitlets
import ipywidgets
import threading
import time
import cv2
import numpy as np
from ..image import bgr8_to_jpeg

class CameraBase(traitlets.HasTraits):
    # OpenCV 처리용 (Numpy)
    value = traitlets.Any()
    # 화면 표시용 (Raw Bytes - Fast Path)
    value_jpeg = traitlets.Any()

    def __init__(self, *args, **kwargs):
        super(CameraBase, self).__init__(*args, **kwargs)
        self._running = False
        self._widget = None
        self._thread = None

    @staticmethod
    def instance(*args, **kwargs):
        raise NotImplementedError

    def widget(self):
        if hasattr(self, '_widget') and self._widget is not None:
            return self._widget

        self._widget = ipywidgets.Image(format='jpeg', width=300, height=300)
        self._start_display_thread()
        return self._widget

    def _start_display_thread(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        """
        화면 갱신 루프:
        이미 압축된 value_jpeg가 있으면 그걸 쓰고(Fast),
        없으면 value를 변환해서 씀(Slow)
        """
        target_fps = 30
        interval = 1.0 / target_fps
        
        while self._running:
            start_time = time.time()
            
            # 1. Fast Path (JPEG 그대로 사용)
            if self.value_jpeg is not None:
                if self._widget is not None:
                    self._widget.value = self.value_jpeg
            
            # 2. Fallback (Numpy -> JPEG 변환)
            elif self.value is not None and self._widget is not None:
                try:
                    self._widget.value = bgr8_to_jpeg(self.value)
                except: pass

            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()