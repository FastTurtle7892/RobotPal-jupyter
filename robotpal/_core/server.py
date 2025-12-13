import asyncio
import websockets
import threading
import cv2
import numpy as np
import json
import struct
import traitlets
from traitlets.config.configurable import SingletonConfigurable

# 서버 포트 설정
PORT_WEB = 9999
PORT_TCP = 9998


class SimulatorServer(SingletonConfigurable):
    """
    C++ 시뮬레이터 통신 서버 (Updated)
    - WebSocket: 수신(Receiver)과 디코딩(Processor)을 분리하여 성능 최적화
    - TCP: Ring Buffer 방식의 패킷 처리로 안정성 강화
    """

    # [Output] 최신 이미지 (BGR format)
    latest_image = traitlets.Any(allow_none=True)

    def __init__(self, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self.loop = None
        self.thread = None
        self.running = False

        # 클라이언트 관리용
        self.active_ws = None
        self.active_tcp_writer = None

        # [NEW] WebSocket 전용 Raw 데이터 큐 (수신/처리 분리)
        # 클래스 멤버로 관리하지만, 실제 생성은 이벤트 루프 내부에서 함
        self.queue_web_raw = None

        self.motor_states = {1: 0.0, 2: 0.0}

        self._start()

    def _start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print(f"[RobotPal] 서버 시작됨 | WebSocket: {PORT_WEB}, TCP: {PORT_TCP}")

    def _run_event_loop(self):
        """asyncio 루프 진입점"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # [NEW] 이벤트 루프 내에서 큐 생성 (Thread-Safety)
        self.queue_web_raw = asyncio.Queue(maxsize=5)

        async def main_runner():
            # 1. 프로세서(디코더) 태스크 시작
            processor_task = asyncio.create_task(self._websocket_processor())

            # 2. 서버 시작
            print(f"[RobotPal] 통신 대기 중... (WS: {PORT_WEB}, TCP: {PORT_TCP})")

            async with websockets.serve(self._handle_ws_receiver, "0.0.0.0", PORT_WEB):
                server_tcp = await asyncio.start_server(self._handle_tcp, '0.0.0.0', PORT_TCP)
                async with server_tcp:
                    # 모든 태스크가 끝날 때까지 대기 (사실상 무한 루프)
                    await asyncio.gather(server_tcp.serve_forever(), processor_task)

        try:
            self.loop.run_until_complete(main_runner())
        except Exception as e:
            print(f"[RobotPal Server Error] {e}")
        finally:
            self.loop.close()

    # ==========================================================
    # [1] WebSocket: Receiver (수신 전용)
    # ==========================================================
    async def _handle_ws_receiver(self, websocket):
        print(f"[WEB] 클라이언트 연결: {websocket.remote_address}")
        self.active_ws = websocket
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # [최적화] 큐가 꽉 차면 오래된 프레임 버리고 최신 것 넣기
                    if self.queue_web_raw.full():
                        try:
                            self.queue_web_raw.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    await self.queue_web_raw.put(message)
                else:
                    print(f"[WEB] 텍스트 메시지: {message}")
                    # 단순 에코 응답 (필요시 삭제 가능)
                    # await websocket.send(f"Server received: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("[WEB] 연결 종료")
        except Exception as e:
            print(f"[WEB] 예외 발생: {e}")
        finally:
            self.active_ws = None

    # ==========================================================
    # [2] WebSocket: Processor (디코딩 전용)
    # ==========================================================
    async def _websocket_processor(self):
        """큐에서 Raw 데이터를 꺼내 디코딩하고 latest_image 업데이트"""
        print("[System] WebSocket 프로세서 시작")
        while True:
            try:
                # 큐에서 데이터 대기
                message = await self.queue_web_raw.get()

                # [프로토콜] 헤더(4바이트) + JPEG 데이터 검증
                if len(message) < 4:
                    continue

                packet_len = struct.unpack('<L', message[:4])[0]
                jpeg_data = message[4:]

                if packet_len != len(jpeg_data):
                    print(f"[WEB] 패킷 길이 불일치! Header: {packet_len}, Actual: {len(jpeg_data)}")
                    # 불일치해도 일단 디코딩 시도해볼 수 있음 (선택사항)

                # 디코딩 및 업데이트
                self._decode_and_update(jpeg_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Processor] 처리 중 에러: {e}")

    # ==========================================================
    # [3] TCP Handler (버퍼링 방식)
    # ==========================================================
    async def _handle_tcp(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"[TCP] 클라이언트 연결됨 ({addr})")
        self.active_tcp_writer = writer

        buffer = bytearray()

        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    print("[TCP] 데이터 수신 중단 (EOF)")
                    break

                buffer.extend(data)

                # 버퍼에 처리 가능한 패킷이 있는지 확인
                while len(buffer) >= 4:
                    # 헤더 파싱 (Little Endian, Unsigned Long)
                    msg_size = struct.unpack('<L', buffer[:4])[0]

                    # 데이터가 아직 다 안 왔으면 대기
                    if len(buffer) < 4 + msg_size:
                        break

                    # 패킷 추출
                    frame_data = buffer[4: 4 + msg_size]

                    # 사용한 데이터 제거
                    buffer = buffer[4 + msg_size:]

                    # 디코딩 및 업데이트
                    self._decode_and_update(frame_data)

        except Exception as e:
            print(f"[TCP] 예외 발생: {e}")
        finally:
            print(f"[TCP] 연결 종료 ({addr})")
            self.active_tcp_writer = None
            writer.close()
            await writer.wait_closed()


    # ==========================================================
    # [4] 공통 로직: 디코딩 & Trait 업데이트
    # ==========================================================
    # def _decode_and_update(self, data_bytes):
    #     nparr = np.frombuffer(data_bytes, np.uint8)
    #     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #     if frame is not None:
    #         # OpenGL 좌표계 대응 상하반전
    #         flipped = cv2.flip(frame, 0)
    #         self.latest_image = flipped
    #     else:
    #         print("[System] 이미지 디코딩 실패")

    # [수정된 코드] Raw Data 처리
    # def _decode_and_update(self, data_bytes):
    #     # 1. 해상도 설정 (SandboxScene.cpp 설정과 반드시 일치해야 함)
    #     H, W = 308, 410 
    #     C = 4  # RGB8 형식이면 3, RGBA8이면 4
        
    #     expected_size = W * H * C
    #     if len(data_bytes) != expected_size:
    #         print(f"[Error] 데이터 크기 불일치! 예상: {expected_size}, 수신: {len(data_bytes)}")
    #         return

    #     try:
    #         # 2. 1차원 바이트 배열을 이미지 형태(H, W, C)로 변환
    #         frame = np.frombuffer(data_bytes, dtype=np.uint8).reshape((H, W, C))
            
    #         # 3. 색상 보정 (OpenGL Raw Data는 보통 RGB 순서, OpenCV는 BGR 사용)
    #         # 만약 색상이 이상하게 보이면 아래 줄을 주석 해제하세요.
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #         # 4. 상하 반전 (OpenGL 좌표계 대응)
    #         flipped = cv2.flip(frame, 0)
    #         self.latest_image = flipped

    #     except Exception as e:
    #         print(f"[System] Raw 디코딩 실패: {e}")

    def _decode_and_update(self, data_bytes):
        """
        수신된 바이트 데이터를 JPEG로 해석하여 OpenCV 이미지로 변환
        """
        try:
            # 1. 바이트 데이터를 numpy 배열(uint8 1D array)로 변환
            # (이 단계에서는 아직 이미지가 아니고 단순 바이트 배열임)
            input_array = np.frombuffer(data_bytes, dtype=np.uint8)

            # 2. cv2.imdecode를 사용하여 JPEG 압축 해제
            # cv2.IMREAD_COLOR: BGR 컬러 이미지로 로드 (투명도 무시)
            # cv2.IMREAD_UNCHANGED: 투명도(Alpha)가 필요하면 이것 사용
            frame = cv2.imdecode(input_array, cv2.IMREAD_COLOR)

            if frame is None:
                # 데이터가 손상되었거나 JPEG 형식이 아닐 경우
                # print("[System] 디코딩 실패 (Frame is None)")
                return

            # 3. 상하 반전 (OpenGL 좌표계 대응)
            # C++에서 TJFLAG_BOTTOMUP을 썼다면 이 과정이 필요 없을 수도 있음.
            # 화면이 거꾸로 보이면 아래 줄을 주석 처리하세요.
            flipped = cv2.flip(frame, 0)
            
            # 4. Traitlets 변수 업데이트 (UI 등 다른 모듈에서 감지)
            self.latest_image = flipped

        except Exception as e:
            print(f"[System] 디코딩 중 예외 발생: {e}")

    # ==========================================================
    # [5] 명령 전송 (Thread-Safe)
    # ==========================================================
    def send_command(self, cmd_dict):
        msg = json.dumps(cmd_dict)
        
        # [수정] 공통 패킷 생성 (헤더 + 데이터)
        data_bytes = msg.encode('utf-8')
        header = struct.pack('<L', len(data_bytes))
        payload = header + data_bytes  # 4바이트 길이 정보 + 실제 JSON 데이터

        # 1. WebSocket 전송 (수정됨)
        if self.active_ws and self.loop:
            asyncio.run_coroutine_threadsafe(
                # 문자열 msg 대신 바이너리 payload를 전송해야 함
                self.active_ws.send(payload), self.loop
            )

        # 2. TCP 전송 (기존 로직 유지하되 payload 변수 활용 가능)
        elif self.active_tcp_writer and self.loop:
            # 기존 코드:
            # data_bytes = msg.encode('utf-8')
            # header = struct.pack('<L', len(data_bytes))
            
            def tcp_send():
                if self.active_tcp_writer and not self.active_tcp_writer.is_closing():
                    self.active_tcp_writer.write(payload) # 위에서 만든 payload 사용

            self.loop.call_soon_threadsafe(tcp_send)

    def update_motor_value(self, channel, value):
        """
        Motor 클래스에서 호출. 특정 채널의 값을 갱신하고 통합 명령 전송
        channel: 1(Left) or 2(Right)
        value: -1.0 ~ 1.0
        """
        self.motor_states[channel] = float(value)

        # 양쪽 모터 값을 합쳐서 JSON 패킷 생성
        payload = {
            "type": "drive",
            "left": self.motor_states[1],
            "right": self.motor_states[2]
        }
        self.send_command(payload)

    def update_servo_value(self, servo_id, angle, speed):
        """
        서보 모터 제어 명령을 C++로 전송
        - servo_id: 모터 ID (1~8)
        - angle: 각도 (-90 ~ 90 또는 0 ~ 180, 시뮬레이터 설정에 따름)
        - speed: 이동 속도
        """
        payload = {
            "type": "servo",
            "id": int(servo_id),
            "angle": float(angle),
            "speed": float(speed)
        }
        self.send_command(payload)



# ==========================================================
# 실행 테스트 (디버그 모드)
# ==========================================================
if __name__ == "__main__":
    import time

    print("=========================================")
    print("   RobotPal Server (v2.0 Updated) Start  ")
    print("=========================================")

    server = SimulatorServer.instance()

    try:
        while True:
            frame = server.latest_image

            if frame is not None:
                cv2.imshow("RobotPal Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        cv2.destroyAllWindows()