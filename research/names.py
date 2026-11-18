from __future__ import annotations

import re
import unicodedata


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def tokens(name: str) -> list[str]:
    return [t for t in fold(name).split() if t and t not in {"jr", "sr"}]


def last_first_initial(name: str) -> tuple[str, str]:
    """Map both 'Novak Djokovic' and 'Djokovic N.' to ('djokovic', 'n')."""
    parts = tokens(name)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    last = parts[-1]
    first = parts[0]
    # tennis-data style: Last F. or Last F.M.
    if len(last) <= 2 and last.isalpha() and len(parts[-2]) > 2:
        return parts[-2], last[0]
    return last, first[0]


def same_player(a: str, b: str) -> bool:
    if fold(a) == fold(b):
        return True
    la, ia = last_first_initial(a)
    lb, ib = last_first_initial(b)
    if not la or not lb or la != lb:
        return False
    return not ia or not ib or ia == ib


def display_key(name: str) -> str:
    last, initial = last_first_initial(name)
    return f"{last}|{initial}" if last else fold(name)
