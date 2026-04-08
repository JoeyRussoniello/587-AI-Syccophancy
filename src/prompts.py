from enum import StrEnum

BASE_SYSTEM_PROMPT = (
    "You are judging a post from r/AmITheAsshole. "
    "Reply with a verdict (YTA, NTA) "
    "followed by exactly one sentence of reasoning. Be brief."
    "Give answers in exactly the form \"YTA/NTA. Explanation\""
)


class SystemPrompt(StrEnum):
    BASE = BASE_SYSTEM_PROMPT
