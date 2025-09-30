import queue
import threading
from stt import SpeechRecognizer
from agent_controller import AgentController
from config import stt_mode

def main():
    q = queue.Queue()

    stt = SpeechRecognizer(input_queue=q)
    agent = AgentController(input_queue=q)

    # Run STT in a separate thread
    if stt_mode["mode"] == "realtime":
        stt_thread = threading.Thread(target=stt.start_realtime, daemon=True)
    elif stt_mode["mode"] == "batch":
        stt_thread = threading.Thread(target=stt.start_batch, daemon=True)
    else:
        raise ValueError(f"Unknown STT mode: {stt_mode['mode']}")
    
    stt_thread.start()

    # Run Agent loop (blocking)
    try:
        agent.process_inputs()
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        stt.stop()
        stt_thread.join()

if __name__ == "__main__":
    main()
