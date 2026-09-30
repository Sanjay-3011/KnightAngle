"""
KnightAngle ADAS Calibration Platform
Phone IMU Bridge & Physical Digital Twin Server

Serves a mobile web interface that captures the smartphone's gyroscope/accelerometer
via HTML5 DeviceOrientation API and streams pitch/roll/yaw over WebSockets in real time.
"""

import asyncio
import json
import socket
import threading
from aiohttp import web

# Shared global state accessible by main.py
latest_imu_data = {
    "connected": False,
    "mode": "floor",     # 'floor' (Error A) or 'mounting' (Error B1)
    "pitch": 0.0,
    "roll": 0.0,
    "yaw": 0.0,
    "tare_pitch": 0.0,
    "tare_roll": 0.0
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>KnightAngle Digital Twin</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #0d0e12; color: #f0f0f0; display: flex; flex-direction: column; align-items: center; justify-content: space-between; min-height: 100vh; padding: 24px 16px; text-align: center; }
        .header { margin-top: 10px; }
        .badge { background: #00d7ff22; color: #00d7ff; border: 1px solid #00d7ff; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 1px; }
        h1 { font-size: 24px; margin-top: 12px; color: #ffffff; letter-spacing: 0.5px; }
        .pfd { width: 220px; height: 220px; border-radius: 50%; border: 2px solid #333a48; position: relative; overflow: hidden; background: #14171f; box-shadow: 0 0 25px rgba(0, 215, 255, 0.15); margin: 20px auto; }
        .horizon { width: 300px; height: 300px; position: absolute; top: -40px; left: -40px; border-top: 2px solid #00ff88; transition: transform 0.05s ease-out; }
        .crosshair { position: absolute; width: 100%; height: 100%; top: 0; left: 0; pointer-events: none; }
        .crosshair::before { content: ''; position: absolute; top: 50%; left: 20%; width: 60%; height: 2px; background: rgba(255,255,255,0.4); }
        .crosshair::after { content: ''; position: absolute; left: 50%; top: 20%; height: 60%; width: 2px; background: rgba(255,255,255,0.4); }
        .telemetry { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%; max-width: 320px; margin: 10px auto; }
        .card { background: #161a23; padding: 14px; border-radius: 12px; border: 1px solid #232a38; }
        .card-label { font-size: 11px; text-transform: uppercase; color: #8892a4; letter-spacing: 1px; }
        .card-val { font-size: 24px; font-weight: 700; margin-top: 4px; color: #00d7ff; }
        .btn-group { display: flex; flex-direction: column; gap: 10px; width: 100%; max-width: 320px; }
        button { background: #00d7ff; color: #000; border: none; padding: 14px; border-radius: 10px; font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.2s; }
        button:active { transform: scale(0.97); }
        .btn-mode { background: #232a38; color: #00d7ff; border: 1px solid #00d7ff; }
        .status { font-size: 12px; color: #8892a4; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">PHYSICAL DIGITAL TWIN</span>
        <h1>KnightAngle Controller</h1>
    </div>

    <div class="pfd">
        <div class="horizon" id="horizon"></div>
        <div class="crosshair"></div>
    </div>

    <div class="telemetry">
        <div class="card">
            <div class="card-label">Pitch Angle</div>
            <div class="card-val" id="valPitch">+0.0°</div>
        </div>
        <div class="card">
            <div class="card-label">Roll Angle</div>
            <div class="card-val" id="valRoll">+0.0°</div>
        </div>
    </div>

    <div class="btn-group">
        <button id="btnTare">CALIBRATE / TARE ZERO</button>
        <button class="btn-mode" id="btnMode">MODE: FLOOR INCLINE (ERROR A)</button>
    </div>

    <div class="status" id="statusTxt">Connecting to KnightAngle Core...</div>

    <script>
        let tareP = 0, tareR = 0;
        let curP = 0, curR = 0;
        let mode = "floor";
        const ws = new WebSocket(`ws://${location.host}/ws`);

        ws.onopen = () => {
            document.getElementById('statusTxt').innerText = "🟢 Connected to KnightAngle Core";
            document.getElementById('statusTxt').style.color = "#00ff88";
        };

        ws.onclose = () => {
            document.getElementById('statusTxt').innerText = "🔴 Disconnected";
            document.getElementById('statusTxt').style.color = "#ff4444";
        };

        async function requestSensors() {
            if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
                try {
                    const resp = await DeviceOrientationEvent.requestPermission();
                    if (resp === 'granted') {
                        window.addEventListener('deviceorientation', handleOrientation);
                    }
                } catch (e) { alert("Sensor permission error: " + e); }
            } else {
                window.addEventListener('deviceorientation', handleOrientation);
            }
        }

        function handleOrientation(e) {
            let beta = e.beta || 0;   // Pitch [-180, 180]
            let gamma = e.gamma || 0; // Roll [-90, 90]

            curP = beta - tareP;
            curR = gamma - tareR;

            document.getElementById('valPitch').innerText = (curP > 0 ? "+" : "") + curP.toFixed(1) + "°";
            document.getElementById('valRoll').innerText = (curR > 0 ? "+" : "") + curR.toFixed(1) + "°";

            const horizon = document.getElementById('horizon');
            horizon.style.transform = `translateY(${curP * 1.5}px) rotate(${curR}deg)`;

            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    pitch: curP,
                    roll: curR,
                    mode: mode
                }));
            }
        }

        document.getElementById('btnTare').onclick = () => {
            requestSensors();
            tareP = curP + tareP;
            tareR = curR + tareR;
        };

        document.getElementById('btnMode').onclick = () => {
            mode = (mode === "floor") ? "mounting" : "floor";
            document.getElementById('btnMode').innerText = (mode === "floor") ? "MODE: FLOOR INCLINE (ERROR A)" : "MODE: MOUNTING ERROR (ERROR B1)";
        };

        // Auto request on touch
        document.body.addEventListener('click', requestSensors, { once: true });
    </script>
</body>
</html>
"""

connected_websockets = set()


async def http_handler(request):
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_websockets.add(ws)
    latest_imu_data["connected"] = True
    print("[PhoneBridge] Smartphone connected as Digital Twin controller!", flush=True)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                latest_imu_data["pitch"] = float(data.get("pitch", 0.0))
                latest_imu_data["roll"] = float(data.get("roll", 0.0))
                latest_imu_data["mode"] = data.get("mode", "floor")
    finally:
        connected_websockets.remove(ws)
        if len(connected_websockets) == 0:
            latest_imu_data["connected"] = False
            print("[PhoneBridge] Smartphone disconnected.", flush=True)

    return ws


def get_local_ip():
    """Retrieves the machine's primary local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def start_phone_bridge_server(port=8080):
    """Starts the Phone IMU bridge server in a background daemon thread."""
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_get('/', http_handler)
        app.router.add_get('/ws', ws_handler)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', port)
        loop.run_until_complete(site.start())
        local_ip = get_local_ip()
        print("\n" + "=" * 60, flush=True)
        print(f"  📱 PHONE DIGITAL TWIN BRIDGE READY!", flush=True)
        print(f"  Open on your smartphone:  http://{local_ip}:{port}", flush=True)
        print("=" * 60 + "\n", flush=True)
        loop.run_forever()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import time
    start_phone_bridge_server(port=8080)
    print("[Standalone] Phone Bridge running. Press Ctrl+C to stop.")
    try:
        while True:
            if latest_imu_data["connected"]:
                p = latest_imu_data["pitch"]
                r = latest_imu_data["roll"]
                m = latest_imu_data["mode"]
                print(f"\r[Phone Telemetry] Mode: {m:<8s} | Pitch: {p:+5.1f}° | Roll: {r:+5.1f}°", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Phone Bridge] Stopped.")
