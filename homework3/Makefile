PROJ = homework3

all: $(PROJ).pdf
$(PROJ).pdf: $(PROJ).tex
	xelatex $(PROJ).tex
	xelatex $(PROJ).tex

q5:
	python question5.py

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.pdf *.fls *.fdb_latexmk *.synctex*

clean_all:
	rm -rf __pycache__ data/*.txt picture/*.png
.PHONY: all clean q5 clean_all