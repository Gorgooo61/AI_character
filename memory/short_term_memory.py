import time
import uuid
from rapidfuzz import fuzz

class ShortTermMemory:
    def __init__(self, retention_seconds=300):
        self.memory = []
        self.retention = retention_seconds

    def add_user_only(self, user_text):
        """Adds a new entry with empty ai field. Returns the entry id."""
        entry_id = uuid.uuid4().hex
        self.memory.append({
            "id": entry_id,
            "timestamp": time.time(),
            "user": user_text,
            "ai": ""
        })
        self.cleanup()
        return entry_id
    
    def set_ai_for_id(self, entry_id, ai_text):
        for m in self.memory:
            if m.get("id") == entry_id:
                m["ai"] = ai_text
                self.cleanup()
                return True

        self.cleanup()
        return False

    def latest_incomplete_id(self):
        """Returns id of the newest entry that has empty ai tag."""
        for i in range(len(self.memory) - 1, -1, -1):
            if not (self.memory[i].get("ai") or "").strip():
                return self.memory[i].get("id")
        return None
    
    def get_entry(self, entry_id):
        for m in self.memory:
            if m.get("id") == entry_id:
                return m
        return None

    def cleanup(self):
        now = time.time()
        self.memory = [m for m in self.memory if now - m["timestamp"] < self.retention]

    def search(self, query, threshold=70, exclude_incomplete_latest=True):
        self.cleanup()
        exclude_id = None
        if exclude_incomplete_latest:
            exclude_id = self.latest_incomplete_id()
        best = None
        best_score = -1
        for m in self.memory:
            if exclude_id is not None and m.get("id") == exclude_id:
                continue
            score = fuzz.partial_ratio(query, m.get("user", ""))
            if score >= threshold and score > best_score:
                best = m
                best_score = score
        return best