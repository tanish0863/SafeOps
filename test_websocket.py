#!/usr/bin/env python3
"""Minimal WebSocket test server"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple HTML test page
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #00ff00; padding: 20px; }
        #log { background: #000; padding: 10px; height: 400px; overflow-y: auto; border: 1px solid #00ff00; }
        .connected { color: #00ff00; }
        .error { color: #ff0000; }
    </style>
</head>
<body>
    <h1>WebSocket Test</h1>
    <button onclick="testWebSocket()">Connect to WebSocket</button>
    <div id="log"></div>
    
    <script>
        let ws = null;
        const log = document.getElementById('log');
        
        function testWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${protocol}//${window.location.host}/ws`;
            log.innerHTML += `Connecting to ${url}<br>`;
            
            try {
                ws = new WebSocket(url);
                ws.onopen = () => {
                    log.innerHTML += '<span class="connected">Connected!</span><br>';
                    ws.send("Hello from client");
                };
                ws.onmessage = (e) => {
                    log.innerHTML += `Received: ${e.data}<br>`;
                };
                ws.onerror = (e) => {
                    log.innerHTML += `<span class="error">Error: ${e}}</span><br>`;
                };
                ws.onclose = () => {
                    log.innerHTML += 'Disconnected<br>';
                };
            } catch (e) {
                log.innerHTML += `<span class="error">Exception: ${e.message}</span><br>`;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    print("WebSocket connection attempt")
    await websocket.accept()
    print("WebSocket accepted")
    try:
        data = await websocket.receive_text()
        print(f"Received: {data}")
        await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import socket
    
    def find_free_port(start=8010):
        port = start
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('0.0.0.0', port))
                s.close()
                return port
            except OSError:
                port += 1
    
    port = find_free_port()
    print(f"Starting WebSocket test server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
