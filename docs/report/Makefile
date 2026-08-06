# M10 discrepancy report — build with:  module load texlive/live2025-gcc-13.3.0 && make
PDF = m10_typo_report.pdf m10_precond_note.pdf
all: $(PDF)
%.pdf: %.tex
	pdflatex -interaction=nonstopmode $< >/dev/null
	pdflatex -interaction=nonstopmode $< >/dev/null
	@echo "built $@"
clean:
	rm -f *.aux *.log *.out *.toc b1.log b2.log build1.log build2.log
.PHONY: all clean
