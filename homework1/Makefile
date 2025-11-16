PROJ = report

all: $(PROJ).pdf
$(PROJ).pdf: $(PROJ).tex
	xelatex $(PROJ).tex
	xelatex $(PROJ).tex

q3:
	python question3.py

q4:
	python question4.py

q5:
	python question5.py

help: 
	@echo "Makefile commands:"
	@echo "  all    - Compile the LaTeX report to PDF"
	@echo "  q3     - Run the script for question 3"
	@echo "  q4     - Run the script for question 4"
	@echo "  q5     - Run the script for question 5"
	@echo "  clean  - Remove auxiliary files generated during compilation"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.pdf *.fls *.fdb_latexmk *.synctex*
.PHONY: all clean q3 q4 q5 help

