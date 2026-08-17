LATEXMK ?= latexmk

.PHONY: all clean distclean

all: paper.pdf

paper.pdf: paper.tex references.bib neurips_2026.sty $(wildcard sections/*.tex) $(wildcard appendix/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf) $(wildcard figures/*.png)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	$(LATEXMK) -c paper.tex

distclean:
	$(LATEXMK) -C paper.tex
