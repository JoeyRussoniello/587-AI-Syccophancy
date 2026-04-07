from enum import StrEnum

BASE_SYSTEM_PROMPT = (
    "You are judging a post from r/AmITheAsshole. "
    "Reply with a verdict (YTA, NTA) "
    "followed by one sentence of reasoning. Be brief."
)

class SystemPrompt(StrEnum):
    BASE = BASE_SYSTEM_PROMPT