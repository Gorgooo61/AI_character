import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    def __init__(self, db_path="data/chroma", collection_name="long_term_memory"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _stable_id(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_fact(self, text):
        text = (text or "").strip()
        embedding = self.embedder.encode([text])[0].tolist()
        #self.collection.add(documents=[text], embeddings=[embedding])
        self.collection.upsert(ids=[self._stable_id(text)], documents=[text], embeddings=[embedding])

    def search(self, query, n_results=1): # not in use, only for test
        q = (query or "").strip()
        embedding = self.embedder.encode([q])[0].tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=max(1, int(n_results)), include=["documents"])
        docs = results.get("documents", [[]])[0]
        if not docs:
            return None
        return docs[0]
    
    def search_with_thresholds(self, query, primary_threshold=0.80, primary_topk=5,
                               fallback_threshold=0.50, fallback_topk=1):
        """
        Thresholded search using cosine distance -> similarity = 1 - distance.

        1) Try candidates with similarity >= primary_threshold; return up to primary_topk.
        2) If none, try similarity >= fallback_threshold; return up to fallback_topk.
        3) If none again, return [].
        """
        q = (query or "").strip()
        q_emb = self.embedder.encode([q])[0].tolist()
        # Request a larger pool so we can filter ourselves
        pool_k = max(primary_topk, fallback_topk, 10)

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=pool_k,
            include=["documents", "distances"]
        )

        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        if not docs or not dists:
            return []

        # Convert distances to cosine similarities: sim=1-dist
        sims = [1.0 - float(d) for d in dists]
        pairs = list(zip(docs, sims))

        # Primary filter
        primary = [p for p in pairs if p[1] >= float(primary_threshold)]
        primary.sort(key=lambda x: x[1], reverse=True)
        if primary:
            return [p[0] for p in primary[:int(primary_topk)]]

        # Fallback filter
        fallback = [p for p in pairs if p[1] >= float(fallback_threshold)]
        fallback.sort(key=lambda x: x[1], reverse=True)
        if fallback:
            return [p[0] for p in fallback[:int(fallback_topk)]]
        # Nothing relevant enough
        return []