PDF=notes_pcg_rkhs_cp.pdf

all: $(PDF)

$(PDF): notes_pcg_rkhs_cp.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<

clean:
	rm -f *.aux *.log *.out
