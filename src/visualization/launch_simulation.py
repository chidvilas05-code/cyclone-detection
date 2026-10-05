"""
Standalone Launcher for 3D Cyclone Movement & Sensory Simulation
Runs a local server and opens the 3D visualization dashboard in your web browser.
"""

import sys
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

PORT = 8055
HTML_FILE = Path(__file__).resolve().parent / "cyclone_3d_simulation.html"


def start_server():
    class SimulationHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ["/", "/index.html", "/simulation"]:
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                with open(HTML_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                super().do_GET()

        def log_message(self, format, *args):
            return  # Suppress routine GET logging for clean console output

    with socketserver.TCPServer(("", PORT), SimulationHandler) as httpd:
        print(f"================================================================")
        print(f"  3D Cyclone Simulation & Multi-Task Forecaster Active!")
        print(f"  URL: http://localhost:{PORT}")
        print(f"  Features:")
        print(f"    - 3D Interactive Earth Map & Rotating Cyclone Vortex")
        print(f"    - Split Right Panel: Live Sensory (Top) vs Predicted (Bottom)")
        print(f"    - Predicted Movement Rendered as Red Dotted Line")
        print(f"================================================================")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    if not HTML_FILE.exists():
        print(f"Error: Simulation HTML not found at {HTML_FILE}")
        sys.exit(1)

    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    webbrowser.open(f"http://localhost:{PORT}")
    try:
        t.join()
    except KeyboardInterrupt:
        print("\nStopping 3D simulation server.")
