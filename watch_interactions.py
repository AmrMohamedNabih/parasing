import time
import os
import sys

def watch_logs():
    log_file = "rag_service.log"
    
    if not os.path.exists(log_file):
        print(f"Waiting for {log_file} to be created...")
        while not os.path.exists(log_file):
            time.sleep(1)
            
    print(f"--- Monitoring {log_file} for LLM Interactions ---")
    
    with open(log_file, "r", encoding="utf-8") as f:
        # Go to the end of the file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            # Highlight special sections
            if "PROMPT SENT TO LLM" in line:
                print("\n" + "█" * 80)
                print("█   NEW LLM INTERACTION DETECTED")
                print("█" * 80)
                print(line.strip())
            elif "AI ANSWER:" in line:
                print("\n" + "✔" * 80)
                print("✔   LLM RESPONSE COMPLETE")
                print("✔" * 80)
                print(line.strip())
            elif "=" * 80 in line or "-" * 80 in line:
                print(line.strip())
            elif "SYSTEM:" in line or "USER:" in line:
                print("\033[94m" + line.strip() + "\033[0m")
            else:
                # Only print lines that seem relevant to the interaction logic
                # to avoid noise from other logs
                if any(k in line for k in ["SYSTEM:", "USER:", "Question:", "Context passages:", "Background Context:"]):
                     print(line.strip())
                elif len(line.strip()) > 0 and not any(k in line for k in ["INFO", "DEBUG", "WARNING", "ERROR"]):
                     print(line.strip())

if __name__ == "__main__":
    try:
        watch_logs()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
