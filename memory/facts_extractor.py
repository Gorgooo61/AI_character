import json
import re

EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise fact extractor.\n"
    "Given a short exchange, extract AT MOST 2 concise, user-centric factual statements.\n"
    "Rules:\n"
    "- Output ONLY a valid JSON array of strings.\n"
    "- No explanations, no markdown, no extra keys.\n"
    "- Facts must be stable and useful later (preferences, constraints, long-term info).\n"
    "- Do NOT include transient details (e.g., 'today', 'right now', temporary moods) unless explicitly long-term.\n"
    "- If there are no useful facts, output an empty JSON array: []\n"
)

EXTRACTION_USER_PROMPT = (
    "Conversation:\n\n"
    "{conversation}\n\n"
    "Extract at most 2 concise facts about the USER that will remain useful later."
)

_USER_TAG_RE = re.compile(r"^\s*\[USER\]\s*:\s*", flags=re.I)

class FactExtractor:

    def __init__(self, generate_callable):
        if not callable(generate_callable):
            raise ValueError("generate_callable must be callable(system_prompt, user_prompt, **kwargs) -> str")
        self.gen = generate_callable

    def _clean_user_text(self, user_text):
        return _USER_TAG_RE.sub("", user_text or "").strip() # removes the [USER] tag

    def _build_conversation_block(self, user_text, assistant_text):
        """
        Builds a consistent conversation snippet for the extractor prompt.
        If user_text is empty (autonomous message), we only include Assistant.
        """
        u = self._clean_user_text(user_text)
        a = (assistant_text or "").strip()

        if u and a:
            return f"User: {u}\nAssistant: {a}"
        if a and not u:
            # Autonomous assistant message: no user input
            return f"Assistant: {a}"
        if u and not a:
            return f"User: {u}"
        return ""

    def extract(self, user_text, assistant_text, max_new_tokens = 120,
                temperature = 0.2, top_p = 0.9):
        """
        Returns a list[str] with at most 2 items.
        """
        conversation = self._build_conversation_block(user_text, assistant_text)
        if not conversation:
            return []

        user_prompt = EXTRACTION_USER_PROMPT.format(conversation=conversation)

        raw = self.gen(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        facts = self._try_parse_json_list(raw)
        if facts is not None:
            return facts

        m = re.search(r"\[[\s\S]*?\]", raw or "")
        if not m:
            return []
        facts = self._try_parse_json_list(m.group(0))
        return facts if facts is not None else []

    def _try_parse_json_list(self, s):
        """
        Parses JSON and validates: list[str], returns up to 2 facts.
        Returns None if parsing/validation fails.
        """
        try:
            x = json.loads((s or "").strip())
        except Exception:
            return None

        if not isinstance(x, list):
            return None

        out = []
        for item in x:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    out.append(t)
            if len(out) >= 2:
                break

        return out