#!/usr/bin/env python3
"""
Simple HTTP server to serve the HTML frontend
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def main():
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    PORT = 8080
    
    # Check if the HTML file exists
    html_file = "simple_frontend.html"
    if not os.path.exists(html_file):
        print(f"❌ Error: {html_file} not found!")
        return 1
    
    # Create a custom handler that serves the HTML file
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.path = '/simple_frontend.html'
            return super().do_GET()
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"🚀 Simple frontend server starting...")
            print(f"📊 Frontend available at: http://localhost:{PORT}")
            print(f"⚠️  Make sure the API server is running on port 8000")
            print(f"   Start API with: python main.py api")
            print(f"   Or: uv run -- python main.py api")
            print()
            print("Press Ctrl+C to stop the server")
            
            # Try to open the browser
            try:
                webbrowser.open(f'http://localhost:{PORT}')
            except:
                pass
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0
    except OSError as e:
        if e.errno == 10048:  # Port already in use on Windows
            print(f"❌ Port {PORT} is already in use. Try a different port or stop the other server.")
        else:
            print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

