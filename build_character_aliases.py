"""Scan books/*.html for speaker names that uniquely identify a single play.

The Shakespeare HTML files mark every dialogue line with a <b>SPEAKER</b> tag.
We collect every distinct speaker per play, invert the map, and keep only the
names that appear in exactly one play's files. Those become extra aliases that
the chat UI can use to route a query to the right play when the embeddings
alone don't carry the proper noun strongly (e.g. "Montague", "Tybalt").

Output: ./character_aliases.json — a JSON object {lowercased_name: work_code}.

Generic role names ("First Soldier", "Messenger", "Servant") fall out naturally
because they appear across many plays, so the unique-to-one-play filter drops
them. A short stoplist catches a few that slip through.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).parent
BOOKS_DIR = PROJECT_DIR / "books"
OUTPUT = PROJECT_DIR / "character_aliases.json"

# Names to discard even if they happen to be unique to one play — too generic
# to use as a play-routing hint.
STOPLIST = {
    "all", "both", "first", "second", "third", "fourth", "fifth",
    "lord", "lady", "servant", "messenger", "soldier", "officer",
    "gentleman", "citizen", "captain", "boy", "girl", "page", "clown",
    "fool", "chorus", "prologue", "epilogue", "old man", "watchman",
    "sailor", "pirate", "drawer", "porter", "ghost", "spirit",
}

# Tags inside <b>...</b> we should skip because they're stage labels, not names.
SKIP_FRAGMENTS = ("scene", " act ", "act ", "exit", "exeunt", "enter ", "re-enter")


def speakers_in_file(path: Path) -> set[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    out: set[str] = set()
    for b in soup.find_all("b"):
        txt = b.get_text(separator=" ", strip=True)
        if not txt or len(txt) < 3:
            continue
        low = txt.lower()
        if any(frag in low for frag in SKIP_FRAGMENTS):
            continue
        # Normalise whitespace, strip trailing punctuation
        txt = re.sub(r"\s+", " ", txt).strip(" .,:;")
        if not txt:
            continue
        out.add(txt)
    return out


def main() -> None:
    if not BOOKS_DIR.exists():
        raise SystemExit(f"books/ not found at {BOOKS_DIR}")

    # work_code -> set of distinct speakers in that play's files
    by_play: dict[str, set[str]] = defaultdict(set)
    for play_dir in sorted(BOOKS_DIR.iterdir()):
        if not play_dir.is_dir():
            continue
        # Skip the Poetry folder — no speakers there, and we don't want to
        # accidentally match sonnet titles as aliases.
        if play_dir.name.lower() == "poetry":
            continue
        for f in play_dir.rglob("*.html"):
            by_play[play_dir.name] |= speakers_in_file(f)

    # Invert: name -> set of work_codes
    name_to_codes: dict[str, set[str]] = defaultdict(set)
    for code, names in by_play.items():
        for n in names:
            name_to_codes[n.lower()].add(code)

    # Keep names that appear in exactly one play AND aren't in the stoplist.
    unique: dict[str, str] = {}
    for name, codes in name_to_codes.items():
        if len(codes) != 1:
            continue
        if name in STOPLIST:
            continue
        # Drop pure numbers and overly-short fragments
        if name.isdigit() or len(name) < 4:
            continue
        unique[name] = next(iter(codes))

    OUTPUT.write_text(json.dumps(unique, indent=2, sort_keys=True))
    print(f"Wrote {len(unique)} unique character aliases to {OUTPUT}")

    # Per-play summary so we can sanity-check the output.
    grouped: dict[str, list[str]] = defaultdict(list)
    for n, c in unique.items():
        grouped[c].append(n)
    for code in sorted(grouped, key=lambda k: -len(grouped[k])):
        sample = sorted(grouped[code])[:8]
        more = "" if len(grouped[code]) <= 8 else f" … +{len(grouped[code]) - 8} more"
        print(f"  {code:18s} {len(grouped[code]):3d}: {', '.join(sample)}{more}")


if __name__ == "__main__":
    main()
