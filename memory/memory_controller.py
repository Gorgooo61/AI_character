from .lore_memory import LoreMemory
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .facts_extractor import FactExtractor

class MemoryController:
    def __init__(
            self,
            generate_callable,
            lore_path="data/lore.json",
            chroma_path="data/chroma",
            chroma_collection="long_term_memory",
            short_retention_seconds=300,
        ):
        self.lore = LoreMemory(path=lore_path)
        self.short = ShortTermMemory(short_retention_seconds)
        self.long = LongTermMemory(db_path=chroma_path, collection_name=chroma_collection)
        self.extractor = FactExtractor(generate_callable)

    def start_turn(self, raw_user_text): # save user prompt to short-term m. first
        raw_user_text = (raw_user_text or "").strip()
        tagged_user = f"[USER]: {raw_user_text}"
        entry_id = self.short.add_user_only(tagged_user) # ai tag empty
        return entry_id

    def build_prompt_with_context(
        self,
        raw_user_text,
        long_primary_threshold=0.80,
        long_primary_topk=5,
        long_fallback_threshold=0.50,
        long_fallback_topk=1,
        lore_threshold=80,
        lore_topk=1,
        short_threshold=70
    ):
        raw_user_text = (raw_user_text or "").strip()
        parts = []

        # USER (always present)
        parts.append(f"[USER]: {raw_user_text}")

        # LORE (fuzzy)
        lore_hits = self.lore.search(raw_user_text, threshold=lore_threshold, topk=lore_topk)
        if lore_hits:
            lore_text = ". ".join([x.strip() for x in lore_hits if x and x.strip()])
            if lore_text:
                parts.append(f"[LORE]: {lore_text}")

        # RECENT (short-term, exclude newest incomplete)
        short_hit = self.short.search(
            raw_user_text,
            threshold=short_threshold,
            exclude_incomplete_latest=True
        )
        if short_hit: # single hit
            recent_user = short_hit.get("user", "").strip()
            recent_ai = short_hit.get("ai", "").strip()
            if recent_user or recent_ai:
                parts.append(f"[RECENT]:\nUser: {recent_user}\nAssistant: {recent_ai}")

        # FACT (long-term, embedding search)
        long_hits = self.long.search_with_thresholds(
            raw_user_text,
            primary_threshold=long_primary_threshold,
            primary_topk=long_primary_topk,
            fallback_threshold=long_fallback_threshold,
            fallback_topk=long_fallback_topk,
        )
        if long_hits:
            fact_text = ". ".join([x.strip() for x in long_hits if x and x.strip()])
            if fact_text:
                parts.append(f"[FACT]: {fact_text}")

        return "\n".join(parts)
    
    # def finalize_turn_and_update_memories(self, short_id, raw_user_text, ai_text):
    #     """
    #     1) Fills the latest short-term placeholder's ai field
    #     2) Runs fact extraction on (latest short-term user, ai)
    #     3) Stores extracted facts into long-term memory
    #     """
    #     ai_text = (ai_text or "").strip()
    #     raw_user_text = (raw_user_text or "").strip()

    #     # Fill short-term AI
    #     self.short.set_ai_for_id(short_id, ai_text)

    #     # Pull the completed entry back (user is tagged)
    #     entry = self.short.get_entry(short_id)
    #     if not entry:
    #         return

    #     user_for_extraction = entry.get("user", "").strip()
    #     ai_for_extraction = entry.get("ai", "").strip()

    #     # Extract facts (max 2)
    #     facts = self.extractor.extract(
    #         user_text=user_for_extraction,
    #         assistant_text=ai_for_extraction
    #     )

    #     for f in facts:
    #         f = (f or "").strip()
    #         if f:
    #             self.long.add_fact(f)

    def extract_and_store_facts(self, user_text, ai_text):
        """
        1) Runs fact extraction on (latest user input, ai)
        2) Stores extracted facts into long-term memory
        """
        user_text = (user_text or "").strip()
        ai_text = (ai_text or "").strip()

        facts = self.extractor.extract(
            user_text=user_text,
            assistant_text=ai_text
        )

        for f in facts:
            f = (f or "").strip()
            if f:
                self.long.add_fact(f)

    # def store_autonomous(self, ai_output):
    #     """Handles autonomous, self-initiated assistant messages (no user query)."""
    #     text = (ai_output or "").strip()
    #     facts = self.extractor.extract("", text)
    #     for f in facts:
    #         f = (f or "").strip()
    #         if f:
    #             self.long.add_fact(f)