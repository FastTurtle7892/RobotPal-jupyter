import os
import sys
import shutil
import tempfile
import threading
import asyncio
import time
import base64
import requests
import aiohttp
from aiohttp import web

# =================================================================
# [1] 환경 감지 및 Display 함수 안전 백업
# =================================================================
try:
    from google.colab import output
    import IPython.display
    from IPython.display import HTML, display as ipy_display, JSON, IFrame
    import ipywidgets
    IS_COLAB = True
    IS_IPYTHON = True
except ImportError:
    IS_COLAB = False
    try:
        import IPython.display
        from IPython.display import IFrame, display as ipy_display
        IS_IPYTHON = True
    except ImportError:
        IS_IPYTHON = False
        def ipy_display(*args, **kwargs): pass

# [핵심] 재실행 안전장치 (Reload-Safe)
# 모듈이 다시 로드되거나 셀이 재실행될 때, 이미 패치된 display를
# 원본으로 착각하고 저장하는 것을 방지합니다.
current_module = sys.modules[__name__]
if not hasattr(current_module, '_original_display'):
    _original_display = IPython.display.display if IS_IPYTHON else print
    setattr(current_module, '_original_display', _original_display)
else:
    _original_display = getattr(current_module, '_original_display')

# =================================================================
# [2] 스마트 디스플레이 (재귀 탐색 & JS Polling)
# =================================================================
def _attach_js_stream(target):
    """
    이미지 위젯 하나에 JS 기반 스트리밍 엔진을 부착합니다.
    """
    # 중복 부착 방지
    if getattr(target, "_is_robotpal_attached", False):
        return
    target._is_robotpal_attached = True
    
    widget_id = id(target)
    target_class = f"robotpal-stream-{widget_id}"
    target.add_class(target_class)
    
    # 파이썬 콜백: 현재 이미지 데이터를 Base64로 변환하여 반환
    callback_name = f"get_frame_{widget_id}"
    def get_frame_callback():
        img_data = target.value
        if img_data:
            b64_str = base64.b64encode(img_data).decode('utf-8')
            return JSON({'b64': b64_str})
        return JSON({'b64': ''})
        
    output.register_callback(callback_name, get_frame_callback)
    
    # JS 코드 주입: 주기적으로 파이썬 콜백을 호출하여 img 태그 src 업데이트
    js_code = f"""
    <script>
    (function() {{
        setTimeout(function() {{
            var wrappers = document.getElementsByClassName('{target_class}');
            if (wrappers.length == 0) return;
            var wrapper = wrappers[0];
            var img = wrapper.querySelector('img');
            
            // 만약 레이아웃 깊숙이 있어서 img를 바로 못 찾으면 재탐색
            if (!img) img = wrapper.getElementsByTagName('img')[0];
            if (!img) return;
            
            var is_running = true;
            function updateFrame() {{
                if (!is_running || !document.body.contains(img)) return;
                
                google.colab.kernel.invokeFunction('{callback_name}', [], {{}})
                .then(function(result) {{
                    if (result.data && result.data['application/json']) {{
                        var b64 = result.data['application/json'].b64;
                        if (b64) img.src = "data:image/jpeg;base64," + b64;
                    }}
                    setTimeout(updateFrame, 33); // 약 30 FPS
                }})
                .catch(function(err) {{
                    setTimeout(updateFrame, 1000);
                }});
            }}
            updateFrame();
            // console.log("📡 Stream Attached: {widget_id}");
        }}, 800); // UI 렌더링 대기 시간
    }})();
    </script>
    """
    ipy_display(HTML(js_code))

def smart_display(*objs, **kwargs):
    """
    display() 호출 시 가로채는 함수.
    1. 원본 display를 호출하여 UI(버튼, 레이아웃 등)를 먼저 그립니다.
    2. 객체 내부를 재귀적으로 탐색하여 '이미지 위젯'을 찾으면 스트리밍을 연결합니다.
    """
    # 1. UI 렌더링 (이게 먼저 실행되어야 버튼이 보입니다)
    _original_display(*objs, **kwargs)
    
    if not IS_COLAB: return

    # 2. 내부 이미지 위젯 탐색 (HBox, VBox 지원)
    def recursive_check(widget):
        # 이미지는 바로 연결
        if isinstance(widget, ipywidgets.Image):
            _attach_js_stream(widget)
        # 컨테이너는 자식들을 탐색
        elif hasattr(widget, 'children'):
            for child in widget.children:
                recursive_check(child)
    
    try:
        for obj in objs:
            if isinstance(obj, ipywidgets.Widget):
                recursive_check(obj)
    except Exception as e:
        print(f"Smart Display Error: {e}")

def apply_patch():
    """시스템의 display 함수를 스마트 버전으로 교체합니다."""
    if IS_COLAB:
        IPython.display.display = smart_display
        print("🚀 [RobotPal] Smart Display Patch Applied (Layout Support)")

# =================================================================
# [3] 브리지 서버 클래스 (스마트 연결 대기)
# =================================================================
class RobotPalBridge:
    def __init__(self, base_url="https://junwoo-seo-1998.github.io/RobotPal/"):
        self.base_url = base_url if base_url.endswith('/') else base_url + '/'
        self.download_dir = os.path.join(tempfile.gettempdir(), "RobotPal_ClientMode")
        self.targets = ["index.html", "RobotPal.js", "RobotPal.wasm", "RobotPal.data", "coi-serviceworker.min.js"]
        self.ws_browser = None
        self.ws_ml = None

    def _setup_files(self):
        if os.path.exists(self.download_dir):
            try: shutil.rmtree(self.download_dir)
            except: pass
        os.makedirs(self.download_dir, exist_ok=True)

        for f in self.targets:
            try:
                r = requests.get(self.base_url + f)
                if r.status_code == 200:
                    content = r.content
                    if f == "index.html":
                        html_str = content.decode('utf-8')
                        patch_script = """
                        <script>
                        (function() {
                            var protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                            var CORRECT_URL = protocol + window.location.host + "/ws";
                            var OriginalWebSocket = window.WebSocket;
                            window.WebSocket = function(url, protocols) {
                                console.log("🔀 Bridge Redirect: " + CORRECT_URL);
                                return new OriginalWebSocket(CORRECT_URL, protocols);
                            };
                        })();
                        </script>
                        """
                        if '<head>' in html_str:
                            html_str = html_str.replace('<head>', '<head>' + patch_script, 1)
                        content = html_str.encode('utf-8')
                    with open(os.path.join(self.download_dir, f), "wb") as file:
                        file.write(content)
            except: pass

    async def _maintain_ml_connection(self):
        ML_URL = "ws://127.0.0.1:9999"
        
        while True:
            # 웹앱이 없으면 연결하지 않고 대기 (데이터 손실 방지)
            if self.ws_browser is None or self.ws_browser.closed:
                await asyncio.sleep(0.5)
                continue

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ML_URL) as ws:
                        self.ws_ml = ws
                        async for msg in ws:
                            if self.ws_browser is None or self.ws_browser.closed: break 
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self.ws_browser.send_str(msg.data)
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                await self.ws_browser.send_bytes(msg.data)
            except: pass
            
            self.ws_ml = None
            await asyncio.sleep(1)

    async def _handler_browser(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_browser = ws
        try:
            async for msg in ws:
                if self.ws_ml and not self.ws_ml.closed:
                    if msg.type == web.WSMsgType.TEXT:
                        await self.ws_ml.send_str(msg.data)
                    elif msg.type == web.WSMsgType.BINARY:
                        await self.ws_ml.send_bytes(msg.data)
        finally:
            self.ws_browser = None
        return ws

    async def _handle_index(self, request):
        return web.FileResponse(os.path.join(self.download_dir, "index.html"))

    async def _on_header(self, request, response):
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cache-Control'] = 'no-store'

    def _run_server_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self._maintain_ml_connection())

        app = web.Application()
        app.on_response_prepare.append(self._on_header)
        app.add_routes([
            web.get('/ws', self._handler_browser),
            web.get('/', self._handle_index),
            web.static('/', self.download_dir)
        ])

        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', 8000)
        loop.run_until_complete(site.start())
        loop.run_forever()

    def start(self):
        # 1. 패치 적용
        apply_patch()
        
        # 2. 파일 준비 및 서버 스레드 시작
        self._setup_files()
        t = threading.Thread(target=self._run_server_thread, daemon=True)
        t.start()
        
        print("\n🚀 [RobotPal Bridge Started]")
        
        # 3. 환경별 화면 띄우기
        if IS_COLAB:
            output.serve_kernel_port_as_iframe(8000, height=800)
        elif IS_IPYTHON:
            print("🔗 Local Link: http://localhost:8000")
            try: ipy_display(IFrame("http://localhost:8000", width='100%', height=800))
            except: pass
        else:
            print("🌐 Open this URL in your browser: http://localhost:8000")

# =================================================================
# [4] 사용자가 호출할 범용 함수
# =================================================================
def start_bridge():
    """RobotPal 시뮬레이터 브리지를 시작합니다."""
    bridge = RobotPalBridge()
    bridge.start()