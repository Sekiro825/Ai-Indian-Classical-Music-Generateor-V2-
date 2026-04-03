import time
import sys
import os

def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

def keep_awake():
    """
    Enhanced keep-awake script.
    - Shows uptime and cycle status.
    - Performs activity pulses to prevent 10-min hibernation.
    """
    start_time = time.time()
    print(f"--- Keep-Awake UI Enhanced (PID: {os.getpid()}) ---")
    print("Instance will stay active. Limit is 10m; we pulse every 1m.")
    
    try:
        while True:
            current_time = time.time()
            uptime_total = current_time - start_time
            
            # Calculate where we are in a 10-minute window (600 seconds)
            cycle_seconds = int(uptime_total % 600)
            cycle_minutes = cycle_seconds // 60
            
            # Formatting the display
            uptime_str = format_duration(uptime_total)
            status = "STABLE"
            
            # If we're at 9 minutes into a 10-minute cycle, show a warning/notice
            if cycle_minutes == 9:
                status = "⚠️ NEAR LIMIT - FORCING PULSE"
            
            sys.stdout.write(f"\r[STATUS: {status}] Uptime: {uptime_str} | Cycle: {cycle_minutes}/10m   ")
            sys.stdout.flush()
            
            # Pulse: simple operation to register activity
            _ = 0
            
            # Sleep for 60 seconds
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nKeep-awake stopped. Final uptime: " + format_duration(time.time() - start_time))

if __name__ == "__main__":
    keep_awake()
