import queue
import time

class AgentController:
    def __init__(self, input_queue: queue.Queue):
        self.input_queue = input_queue

    def process_inputs(self):
        """Continuously check for new STT inputs and handle them."""
        while True:
            try:
                item = self.input_queue.get(timeout=1)
                print(f"[Agent] ({time.strftime('%H:%M:%S')}) User said: {item['text']}")
            except queue.Empty:
                # no new input
                continue
