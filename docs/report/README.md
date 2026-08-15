# Report to S. Danilov — mEVP divergence form and wide halo

Two documents, both built from `/work/ab0995/a270088/port2/m9/m9_results.json`:

| file | what |
|---|---|
| `onepager.pdf` | one page: what pays, what it costs, what does not pay. Start here. |
| `danilov_mevp_report.pdf` | the full note, 8 pages incl. 3 figures |

Rebuild:

    python make_report_figs.py     # fig1/fig2/fig3 from the results JSON
    export PATH=/sw/spack-levante/texlive-live2025-3r2myy/bin/x86_64-linux:$PATH
    pdflatex onepager.tex ; pdflatex danilov_mevp_report.tex   # twice for refs

🔴 **Every speed number is quoted at that mesh's OPERATING POINT** — the largest node count at
which the model still gains from more GPUs, read off the measured strong-scaling curve of the
standard scheme (CORE2 1 node, fArc 4, DARS 8, NG5 16). This is not a detail: at a common
16 nodes the same options measure roughly twice as well, because CORE2 on 16 nodes is 45 % slower
than on 2 and the ice exchange is then a much larger share of a step nobody would pay for. Runs
`op_core2_g4`, `op_farc_g16`, `op_dars_g32`, `rep_ng5_g64` — one allocation each, all six
configurations back to back.

Deliberately NOT the campaign's own figures (`scripts/m9_figs.py`): those use the study's internal
shorthand for the scheme variants, which means nothing to a reader outside the project.
