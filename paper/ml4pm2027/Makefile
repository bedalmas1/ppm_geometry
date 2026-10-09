# Requires a TeX distribution (TeX Live or MiKTeX) with latexmk + Perl.
.PHONY: all assets pdf final clean
all: assets pdf
assets:
	cd ../.. && uv run python paper/ml4pm2027/scripts/make_assets.py
pdf:
	latexmk main.tex
final:
	latexmk submission.tex
clean:
	latexmk -C main.tex && latexmk -C submission.tex
