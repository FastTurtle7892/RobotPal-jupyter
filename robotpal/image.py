import cv2
import traitlets


def bgr8_to_jpeg(value, quality=75):
    """
    Numpy 배열(BGR)을 입력받아 JPEG 바이너리로 변환하는 함수
    (위젯 표시용 transform 함수)
    """
    if value is None:
        return bytes()

    return cv2.imencode('.jpg', value)[1].tobytes()