import asyncio
import websockets
import threading
import cv2
import numpy as np
import json
import struct
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from concurrent.futures import ThreadPoolExecutor

# 서버 포트 설정
PORT_WEB = 9999
PORT_TCP = 9998


class SimulatorServer(SingletonConfigurable):
    """
    C++ 시뮬레이터 통신 서버 (ISP Optimized)
    - 최적화 적용: ThreadPool, Zero-Copy JPEG Pass-through, AI Resizing
    """

    # [Output 1] AI용: 224x224 리사이징된 Numpy (가벼움)
    latest_image = traitlets.Any(allow_none=True)
    
    # [Output 2] 화면용: 원본 JPEG 바이너리 (빠름)
    latest_jpeg = traitlets.Bytes(allow_none=True)

    def __init__(self, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self.loop = None
        self.thread = None
        self.running = False

        self.active_ws = None
        self.active_tcp_writer = None
        self.queue_web_raw = None

        self.motor_states = {1: 0.0, 2: 0.0}
        
        # [최적화] 디코딩 전용 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=4)

        self._start()

    def _start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print(f"[RobotPal] 서버 시작됨 | WebSocket: {PORT_WEB}, TCP: {PORT_TCP}")

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # [최적화] 큐 사이즈 1 (실시간성 보장)
        self.queue_web_raw = asyncio.Queue(maxsize=1)

        async def main_runner():
            processor_task = asyncio.create_task(self._websocket_processor())
            print(f"[RobotPal] 통신 대기 중... (WS: {PORT_WEB}, TCP: {PORT_TCP})")

            async with websockets.serve(self._handle_ws_receiver, "0.0.0.0", PORT_WEB):
                server_tcp = await asyncio.start_server(self._handle_tcp, '0.0.0.0', PORT_TCP)
                async with server_tcp:
                    await asyncio.gather(server_tcp.serve_forever(), processor_task)

        try:
            self.loop.run_until_complete(main_runner())
        except Exception as e:
            print(f"[RobotPal Server Error] {e}")
        finally:
            self.loop.close()

    # ==========================================================
    # [1] WebSocket: Receiver
    # ==========================================================
    async def _handle_ws_receiver(self, websocket):
        self.active_ws = websocket
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if self.queue_web_raw.full():
                        try: self.queue_web_raw.get_nowait()
                        except: pass
                    await self.queue_web_raw.put(message)
        except: pass
        finally: self.active_ws = None

    # ==========================================================
    # [2] WebSocket: Processor (ISP 로직 적용)
    # ==========================================================
    async def _websocket_processor(self):
        print("[System] WebSocket 프로세서 시작")
        loop = asyncio.get_running_loop()

        while True:
            try:
                message = await self.queue_web_raw.get()
                if len(message) < 4: continue

                packet_len = struct.unpack('<L', message[:4])[0]
                jpeg_data = message[4:]

                # [최적화 A] 화면 표시용: 원본 JPEG 즉시 업데이트 (디코딩 X)
                self.latest_jpeg = bytes(jpeg_data)

                # [최적화 B] AI용: 백그라운드 스레드에서 리사이즈 수행
                frame = await loop.run_in_executor(
                    self.executor, self._decode_and_resize, jpeg_data
                )
                if frame is not None:
                    self.latest_image = frame

            except asyncio.CancelledError: break
            except Exception: pass

    # ==========================================================
    # [3] TCP Handler
    # ==========================================================
    async def _handle_tcp(self, reader, writer):
        self.active_tcp_writer = writer
        loop = asyncio.get_running_loop()
        buffer = bytearray()

        try:
            while True:
                data = await reader.read(4096)
                if not data: break
                buffer.extend(data)

                while len(buffer) >= 4:
                    msg_size = struct.unpack('<L', buffer[:4])[0]
                    if len(buffer) < 4 + msg_size: break

                    frame_data = buffer[4: 4 + msg_size]
                    
                    # [최적화 A]
                    self.latest_jpeg = bytes(frame_data)
                    
                    # [최적화 B]
                    frame = await loop.run_in_executor(
                        self.executor, self._decode_and_resize, frame_data
                    )
                    if frame is not None:
                        self.latest_image = frame

                    buffer = buffer[4 + msg_size:]
        except: pass
        finally:
            self.active_tcp_writer = None
            writer.close()

    # ==========================================================
    # [4] 공통 로직: 디코딩 & 리사이즈 (CPU Bound)
    # ==========================================================
    # ==========================================================
    def _decode_and_resize(self, data_bytes):
        """AI를 위해 224x224로 리사이즈"""
        try:
            nparr = np.frombuffer(data_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # 1. 상하반전
                flipped = cv2.flip(frame, 0)
                # 2. 리사이즈 (JetBot 하드웨어 흉내)
                return cv2.resize(flipped, (224, 224), interpolation=cv2.INTER_LINEAR)
        except: pass
        return None
        # ==========================================================
    # def _decode_and_resize(self, data_bytes):
    #     """AI를 위해 224x224로 리사이즈"""
    #     try:
    #         nparr = np.frombuffer(data_bytes, np.uint8)
    #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #         if frame is not None:
    #             # 1. 상하반전
    #             flipped = cv2.flip(frame, 0)
    #             # 2. 리사이즈 (JetBot 하드웨어 흉내)
    #             return flipped
    #     except: pass
    #     return None

    # ==========================================================
    # [5] 명령 전송
    # ==========================================================
    def send_command(self, cmd_dict):
        msg = json.dumps(cmd_dict)
        if self.active_ws and self.loop:
            payload = msg.encode('utf-8')
            header = struct.pack('<I', len(payload))
            asyncio.run_coroutine_threadsafe(
                self.active_ws.send(header + payload), self.loop
            )
        elif self.active_tcp_writer and self.loop:
            data_bytes = msg.encode('utf-8')
            header = struct.pack('<L', len(data_bytes))
            def tcp_send():
                if self.active_tcp_writer:
                    self.active_tcp_writer.write(header + data_bytes)
            self.loop.call_soon_threadsafe(tcp_send)

    def update_motor_value(self, channel, value):
        self.motor_states[channel] = float(value)
        self.send_command({"type": "drive", "left": self.motor_states[1], "right": self.motor_states[2]})

    def update_servo_value(self, servo_id, angle, speed):
        self.send_command({"type": "servo", "id": int(servo_id), "angle": float(angle), "speed": float(speed)})


if __name__ == "__main__":
    import time
    server = SimulatorServer.instance()
    try:
        while True:
            # 테스트 확인용
            frame = server.latest_image
            if frame is not None:
                cv2.imshow("RobotPal Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()