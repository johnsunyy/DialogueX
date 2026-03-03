import socket
import os
import re
import subprocess
import time
import sys
import webbrowser

def get_local_ip():
    """
    Detects the local IP address associated with the default route.
    """
    try:
        # connect to an external server to get the ip reachable by others
        # We don't actually send data, just use it to determine the interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def update_frontend_config(ip_address):
    """
    Updates the BACKEND_URL in frontend/index.html to match the detected IP.
    """
    frontend_path = os.path.join("frontend", "index.html")
    if not os.path.exists(frontend_path):
        print(f"Error: {frontend_path} not found!")
        return False
    
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex to find const BACKEND_URL = "http://...:5000";
        # We look for the variable definition and replace the IP part
        # Pattern handles: const BACKEND_URL = "http://<any_ip>:5000";
        pattern = r'const BACKEND_URL = "http://[^:]+:5000";'
        replacement = f'const BACKEND_URL = "http://{ip_address}:5000";'
        
        new_content = re.sub(pattern, replacement, content)
        
        if content == new_content:
            print(f"✅ Frontend configuration already matches IP: {ip_address}")
        else:
            with open(frontend_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"✅ Updated frontend/index.html with IP: {ip_address}")
            
        return True
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir) # Ensure we are in the project root
    
    print("\n" + "="*50)
    print("   DIALOGUE-X UNIVERSAL LAUNCHER")
    print("="*50)
    
    # 1. Detect IP
    ip = get_local_ip()
    print(f"📡 Detected Local IP: {ip}")
    
    # 2. Update Config
    if not update_frontend_config(ip):
        print("Failed to update configuration. Exiting.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("\n" + "-"*50)
    print(f"🚀 Starting Servers...")
    print(f"Backend:  http://{ip}:5000")
    print(f"Frontend: http://{ip}:8080")
    print("-"*50 + "\n")

    try:
        # 3. Start Backend
        backend_path = os.path.join(root_dir, "backend")
        backend_process = subprocess.Popen([sys.executable, "server.py"], cwd=backend_path)
        
        # Give backend a moment to initialize
        time.sleep(3)
        
        # 4. Start Frontend
        frontend_path = os.path.join(root_dir, "frontend")
        frontend_process = subprocess.Popen([sys.executable, "-m", "http.server", "8080", "--bind", "0.0.0.0"], cwd=frontend_path)
        
        # 5. Open Browser
        print("Opening browser...")
        time.sleep(1)
        webbrowser.open(f"http://{ip}:8080")
        
        print("\n✅ System running! Press Ctrl+C to stop.\n")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Check if processes are still alive
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly.")
                break

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping servers...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        if 'backend_process' in locals() and backend_process.poll() is None:
            backend_process.terminate()
        if 'frontend_process' in locals() and frontend_process.poll() is None:
            frontend_process.terminate()
        print("See you next time!")

if __name__ == "__main__":
    main()
