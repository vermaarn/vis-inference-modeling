import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


# APE webservice endpoint; see documentation:
# `https://attempto.ifi.uzh.ch/site/docs/ape_webservice.html`
APE_WEBSERVICE_URL = "http://attempto.ifi.uzh.ch/ws/ape/apews.perl"


def load_ace_sentences(json_path: Path) -> str:
    """
    Load ACE sentences from a JSON file and concatenate them into a single ACE text.

    The JSON is expected to have an `ace_sentences` field that is a list of strings,
    as in `ace_comments/181/1.json`.
    """
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sentences = payload.get("ace_sentences") or []
    if not isinstance(sentences, list):
        raise ValueError(f"`ace_sentences` must be a list in {json_path}")

    # Join sentences into one ACE text. APE accepts multi-sentence input.
    ace_text = " ".join(s.strip() for s in sentences if s and s.strip())
    if not ace_text:
        raise ValueError(f"No ACE sentences found in {json_path}")

    return ace_text


def ace_to_drs(ace_text: str, solo: str = "drspp") -> str:
    """
    Send ACE text to the Attempto Parsing Engine (APE) webservice and
    return the DRS representation.

    `solo` controls which single output is returned by APE.
    Common options include:
      - 'drs'   : DRS as a Prolog term
      - 'drspp' : pretty-printed DRS (human-readable)
      - 'drsxml': DRS in XML
    """
    if not ace_text.strip():
        raise ValueError("ACE text is empty")

    params = {
        "text": ace_text,
        "solo": solo,
        # Let APE guess some interpretation details; matches examples in docs.
        "guess": "on",
    }

    query = urllib.parse.urlencode(params)
    url = f"{APE_WEBSERVICE_URL}?{query}"

    with urllib.request.urlopen(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def main(argv: list[str]) -> int:
    """
    Example script:
    - Reads ACE sentences from `ace_comments/181/1.json` by default
      (or from a JSON path passed as the first CLI argument),
    - Sends them to APE,
    - Prints the ACE text and the resulting DRS.
    """
    if len(argv) > 1:
        json_path = Path(argv[1])
    else:
        # Default example provided by the user
        json_path = Path(__file__).parent / "ace_comments" / "181" / "1.json"

    try:
        ace_text = load_ace_sentences(json_path)
    except Exception as exc:
        print(f"Failed to load ACE sentences from {json_path}: {exc}", file=sys.stderr)
        return 1

    print("=== ACE text ===")
    print(ace_text)
    print()

    print("=== DRS from APE (solo=drspp) ===")
    try:
        drs_output = ace_to_drs(ace_text, solo="drspp")
    except Exception as exc:
        print(f"Failed to obtain DRS from APE webservice: {exc}", file=sys.stderr)
        return 1

    print(drs_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

