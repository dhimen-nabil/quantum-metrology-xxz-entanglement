# Usage:  make all   (figures, validation, PDFs)
.PHONY: all figures validate pdf clean

all: figures validate pdf

figures:
	python scripts/closed_system_figures.py figures
	python scripts/dephasing_figures.py .

validate:
	python scripts/validate.py

pdf:
	cd report && pdflatex main.tex && pdflatex main.tex
	cd paper && pdflatex paper.tex && pdflatex paper.tex

clean:
	rm -f report/*.aux report/*.log report/*.out report/*.toc paper/*.aux paper/*.log paper/*.out
