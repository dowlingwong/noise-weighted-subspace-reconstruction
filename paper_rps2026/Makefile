LATEXMK ?= latexmk

.PHONY: all clean distclean

all: paper.pdf

figures/classes.pdf: figures/classes.tex
	cd figures && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error classes.tex

paper.pdf: figures/classes.pdf paper.tex references.bib neurips_2026.sty $(wildcard sections/*.tex) $(wildcard appendix/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf) $(wildcard figures/*.png)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	$(LATEXMK) -c paper.tex

distclean:
	$(LATEXMK) -C paper.tex
