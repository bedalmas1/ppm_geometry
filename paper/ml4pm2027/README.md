# ML4PM 2027 paper draft

LaTeX draft of the workshop paper for ICPM 2027 / ML4PM, built from this repository's research project ([`PLAN.md`](../../PLAN.md), [`STATUS.md`](../../STATUS.md)).

**Current scope (2026-10-08):** the workshop paper covers three research questions, chosen by the user to fit the 12-page CFP limit. RQ1–RQ3 and H1–H3 in the paper are renumbered from the spec; the map is in a comment at the top of `sections/introduction.tex`.

| Paper | Spec | Topic | Hypotheses |
|---|---|---|---|
| RQ1 | RQ1, reframed | what the models learn (descriptive) | none |
| RQ2 | RQ3 | reliability: errors beyond confidence | H1 = spec H4 |
| RQ3 | RQ5 | future organisation | H2 = spec H6, H3 = spec H7 |

RQ1 was reframed on 2026-10-08 from "does the objective induce systematically different geometry?" to "what do PPM models learn?". The old wording could not be answered with only one controlled pair. Objective differences are now a secondary contrast inside RQ1, attributed only when B1 vs. B2 shows them. Spec H1/H2 are no longer formal hypotheses. Spec RQ2 and RQ4 (and H3, H5) are deferred to a journal extension. Introduction, Formulation, Related Work, Methodology and Setup are written. Results, Discussion/Threats and Conclusion are `\TODO{}` stubs with page budgets, with no numbers and no hypothesis verdicts.

## Requirements

| Tool | Tested with | Notes |
|---|---|---|
| TeX distribution | MiKTeX 25.12 (pdfTeX 4.23) | TeX Live 2023+ also works. MiKTeX installs missing packages on first build. |
| `latexmk` + Perl | latexmk 4.88, Git-Bash Perl | Strawberry Perl also works on Windows. |
| BibTeX | bundled | Uses `splncs04.bst`, which is included here. |
| Python env | repo `.venv` (`uv sync`) | Only needed to regenerate `generated/` tables. |

LaTeX packages used: `llncs` (class bundled here), `amsmath`, `amssymb`, `booktabs`, `multirow`, `graphicx`, `xcolor`, `tikz` (libraries: `arrows.meta`, `calc`, `angles`, `quotes`, …), `xspace`, `enumitem`, `hyperref`, `cleveref`, `todonotes`, `microtype`, `lmodern`.

## Build

```powershell
# from paper/ml4pm2027/
.\build.ps1              # regenerate tables, then compile the draft -> build\main.pdf
.\build.ps1 -Final       # submission version, TODO boxes hidden    -> build\submission.pdf
.\build.ps1 -SkipAssets  # compile only
```

```bash
make              # assets + draft pdf   (Linux/macOS, or Git Bash with make)
make final        # submission pdf (no TODO boxes)
latexmk main.tex  # draft pdf only
```

PDFs are written to `build/`, which is gitignored. The draft shows orange `\TODO{}` boxes (work to do) and blue `\QUESTION{}` boxes (parked decisions, mirrored in PLAN.md "Parked options"). Both are hidden in the submission build. `submission.tex` is a two-line wrapper that defines `\FINAL` and inputs `main.tex`. Always check the page count on `build/submission.pdf`.

You can also build on Overleaf: upload the folder and set the main file to `main.tex`. The `generated/` tables are committed, so no Python step is needed there.

## Layout

```
main.tex              preamble, notation macros, section includes
sections/*.tex        one file per section
figures/fig_concept.tex   Fig. 1 (pure TikZ schematic, no data)
generated/            tables written by scripts/make_assets.py -- never edit by hand
scripts/make_assets.py    reads data/processed/*/manifest.json -> generated/tab_datasets.tex
scripts/normalize_bib.py  one-off BibTeX clean-up (see docstring)
references.bib        every entry fetched programmatically (DOI -> CrossRef/DataCite, arXiv API)
llncs.cls, splncs04.bst   Springer LNCS class and bibliography style
```

## Rules for extending this draft

- **Numbers come from scripts.** Any table, figure or in-text number must be generated from `results/` or `data/processed/` by a script in `scripts/` (PLAN.md Phase 11). Do not copy numbers by hand.
- **Citations are fetched, not typed.** Add BibTeX only via DOI content negotiation or the arXiv BibTeX endpoint, then run `scripts/normalize_bib.py`.
- **Keep the families separate.** Family A and Family B results go in separate tables and figures, and H1/H2 verdicts are drawn from Family B only (spec §5).
- **No projections as evidence.** UMAP and t-SNE may appear only as illustrations, never as quantitative evidence (spec §7, §14).
- **Respect the venue limits.** The ICPM 2027 workshop CFP (confirmed 2026-10-07) says: submit via EasyChair (`icpm2027`); English; original work; **at most 12 pages including figures, bibliography and appendices**; a short abstract that states the relation to the workshop topics, the problem, the goal, the results achieved and the relation to the literature; proceedings in Springer **LNBIP** (LNCS format, as used here); at least one author must present; LNBIP acceptance rate at most 40%. Appendices count toward the 12 pages, so supplementary detail belongs in the released artefact, not an appendix.
- **Current length:** `submission.pdf` is 9 pages: the body ends early on page 7, and references take about 1.8 pages. That leaves about 3.5–4 pages for Results (about 3.3), Discussion/Threats (about 0.6) and Conclusion (about 0.2). The budgets are noted in each stub.
