import re

DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

WORD_TO_DIGIT = {v: k for k, v in DIGIT_TO_WORD.items()}
WORD_TO_DIGIT["niner"] = "9"


def normalize_for_wer(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[,\.!?;:\"\(\)]", " ", t)
    t = re.sub(r"\bniner\b", "nine", t)

    def expand_fl(m):
        digits = m.group(1)
        return "flight level " + " ".join(DIGIT_TO_WORD[d] for d in digits)
    t = re.sub(r"\bfl\s*(\d{2,3})\b", expand_fl, t)

    def expand_rwy(m):
        digits = m.group(1)
        side_map = {"l": "left", "r": "right", "c": "center"}
        side = side_map.get(m.group(2).lower(), m.group(2)) if m.group(2) else ""
        words = " ".join(DIGIT_TO_WORD[d] for d in digits)
        return ("runway " + words + (" " + side if side else "")).strip()
    t = re.sub(r"\brwy\s*(\d{1,2})([lrc]?)\b", expand_rwy, t)

    t = re.sub(r"\b(\d+)\b", lambda m: " ".join(DIGIT_TO_WORD[d] for d in m.group(1)), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_display(text: str) -> str:
    t = text.strip()

    def compress_fl(m):
        digits = re.findall(r"\b(" + "|".join(WORD_TO_DIGIT.keys()) + r")\b", m.group(1))
        if digits:
            return "FL" + "".join(WORD_TO_DIGIT[d] for d in digits)
        return m.group(0)
    t = re.sub(
        r"\bflight level\s+([a-z ]+?)(?=\s+[a-z]+\b|$)",
        compress_fl, t, flags=re.IGNORECASE
    )

    def compress_rwy(m):
        digits = re.findall(r"\b(" + "|".join(WORD_TO_DIGIT.keys()) + r")\b", m.group(1))
        side_map = {"left": "L", "right": "R", "center": "C"}
        side = side_map.get(m.group(2).lower(), "") if m.group(2) else ""
        if digits:
            return "RWY" + "".join(WORD_TO_DIGIT[d] for d in digits) + side
        return m.group(0)
    t = re.sub(
        r"\brunway\s+([a-z ]+?)\s*(left|right|center)?\b",
        compress_rwy, t, flags=re.IGNORECASE
    )

    t = re.sub(r"(\d+)\s+decimal\s+(\d+)", r"\1.\2", t, flags=re.IGNORECASE)
    return t.strip()


if __name__ == "__main__":
    tests_wer = [
        "SWISS225, descend FL180!",
        "contact 127.4",
        "squawk 4567",
        "niner thousand feet",
    ]
    tests_display = [
        "descend flight level one eight zero",
        "contact one two seven decimal four",
        "squawk four five six seven",
        "runway two eight left",
    ]

    for t in tests_wer:
        print(f"{t!r} -> {normalize_for_wer(t)!r}")

    for t in tests_display:
        print(f"{t!r} -> {normalize_for_display(t)!r}")
