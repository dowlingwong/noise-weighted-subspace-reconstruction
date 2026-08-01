.PHONY: setup-template pdf clean

setup-template:
	sh scripts/fetch_template.sh

pdf: setup-template
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C

