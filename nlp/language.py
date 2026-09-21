import re


def detect_language(text: str) -> str:
    clean = text.lower().strip()
    if not clean:
        return "unknown"

    hindi_chars = bool(re.search(r"[\u0900-\u097F]", clean))
    english_words = len(re.findall(r"\b[a-zA-Z]{3,}\b", clean))
    hindi_words = len(re.findall(r"[\u0900-\u097F]{2,}", clean))

    if hindi_chars and hindi_words > 0 and english_words == 0:
        return "hindi"
    if hindi_chars and english_words > 0:
        return "hinglish"
    if english_words > 0:
        return "english"
    return "unknown"
