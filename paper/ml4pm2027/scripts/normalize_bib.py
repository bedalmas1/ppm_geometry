"""One-off normaliser for references.bib (entries fetched from CrossRef/DataCite/arXiv).

- removes `url` when a DOI is present (splncs04 prints both otherwise);
- wraps titles in an extra pair of braces so splncs04 keeps acronyms/case
  (LSTM, BPI, LUPIN, PPM, ...) exactly as the publisher metadata gives them;
- drops the placeholder `pages={1--1}` CrossRef returns for early-access articles.
Content (authors, venues, years, DOIs) is never altered.
"""
import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "references.bib"
s = p.read_text(encoding="utf-8")
entries = re.split(r"(?=^@)", s, flags=re.M)
out = []
for e in entries:
    if e.startswith("@") and re.search(r"\bdoi\s*=", e, re.I):
        e = re.sub(r"\s*\burl\s*=\s*\{[^}]*\},?", "", e)
    e = re.sub(r"(\btitle\s*=\s*)\{(?!\{)(.*?)\}(\s*,)", r"\1{{\2}}\3", e, count=1)
    e = re.sub(r",?\s*pages\s*=\s*\{1--1\}", "", e)
    out.append(e)
p.write_text("".join(out), encoding="utf-8")
print("normalised", sum(1 for e in out if e.startswith("@")), "entries")
