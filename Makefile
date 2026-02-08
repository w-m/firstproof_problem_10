PDFS=notes_pcg_rkhs_cp.pdf answer_pcg_rkhs_cp.pdf proof.pdf final_proof.pdf

all: $(PDFS)

notes_pcg_rkhs_cp.pdf: notes_pcg_rkhs_cp.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<

answer_pcg_rkhs_cp.pdf: answer_pcg_rkhs_cp.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<

proof.pdf: proof.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<

final_proof.pdf: final_proof.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<

clean:
	rm -f *.aux *.log *.out
