PROJ = report

all: $(PROJ).pdf
$(PROJ).pdf: $(PROJ).tex
	xelatex $(PROJ).tex
	xelatex $(PROJ).tex

q1:
	export QT_QPA_PLATFORM=xcb
	python question1.py

q4:
	export QT_QPA_PLATFORM=xcb
	python question4.py

help: 
	@echo "Makefile commands:"
	@echo "  all    - Compile the LaTeX report to PDF"
	@echo "  q1     - Run the script for question 1"
	@echo "  q4     - Run the script for question 4"
	@echo "  clean  - Remove tex files generated during compilation"
	@echo "  clean_all  - Remove all files generated during compilation"


clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.pdf *.fls *.fdb_latexmk *.synctex*

clean_all:
	rm -rf __pycache__ data/*.txt picture/*.png
.PHONY: all clean q1 q4 help clean_all

