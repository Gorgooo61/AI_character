import json
from rapidfuzz import fuzz

class LoreMemory:
    def __init__(self, path="data/lore.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.lore_data = json.load(f)

    def search(self, query, threshold=80, topk=1):
        q = (query or "").strip()
        scored = []
        for t in self.lore_data:
            s = fuzz.partial_ratio(q, t)
            if s >= threshold:
                scored.append((t, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:int(topk)]]