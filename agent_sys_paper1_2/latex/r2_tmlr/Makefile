.PHONY: setup-template pdf anonymity-check clean

setup-template:
	sh scripts/fetch_template.sh

pdf: setup-template
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

anonymity-check:
	sh scripts/check_anonymity.sh

clean:
	latexmk -C

