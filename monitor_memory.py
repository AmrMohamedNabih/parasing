#!/usr/bin/env python3
"""
Monitor memory usage of the Flask app and OCR engines.
Run this while Flask is running to see real-time memory usage.
"""

import psutil
import time
import subprocess

def get_process_memory_macos(pid):
    """Get memory usage using macOS ps command."""
    try:
        # Use ps command which is more reliable on macOS
        result = subprocess.run(
            ['ps', '-o', 'rss=,vsz=', '-p', str(pid)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            output = result.stdout.strip().split()
            if len(output) >= 2:
                rss_kb = float(output[0])  # RSS in KB
                vsz_kb = float(output[1])  # VSZ in KB
                return {
                    'rss_mb': rss_kb / 1024,
                    'vsz_mb': vsz_kb / 1024
                }
    except Exception as e:
        pass
    return None

def find_flask_process():
    """Find the Flask app process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'app.py' in ' '.join(cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_memory(interval=2):
    """Monitor memory usage continuously."""
    print("=" * 60)
    print("Flask App Memory Monitor (macOS)")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            pid = find_flask_process()
            if pid:
                mem = get_process_memory_macos(pid)
                if mem:
                    print(f"PID: {pid:5d} | "
                          f"RAM: {mem['rss_mb']:7.1f} MB | "
                          f"Virtual: {mem['vsz_mb']:7.1f} MB")
                else:
                    print(f"PID: {pid:5d} | Unable to read memory")
            else:
                print("Flask app not running...")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_memory()
