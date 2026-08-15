#!/usr/bin/env python3
"""Figures for the report to S. Danilov on the divergence-form / wide-halo mEVP work.

Deliberately NOT the campaign's internal figures (scripts/m9_figs.py): those carry the study's
own shorthand for the scheme variants, which means nothing outside the project. Here every
series is named the way the report names it.

All numbers come from /work/.../m9/m9_results.json, i.e. from the same measured runs; nothing
is entered by hand except the accuracy table, which comes from the 1-year runs analysed by
scripts/m9_accuracy.py and is quoted in the report text as well.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = "/work/ab0995/a270088/port2/m9/m9_results.json"
OUT = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(J))

C_LAG8, C_LAG4 = "#c0392b", "#e8896b"
C_WIDE, C_WDIV = "#1f6fb4", "#7fb3dd"
GREY = "#555555"


def run(tag):
    for r in D["runs"]:
        if r.get("tag") == tag:
            return r
    return None


def pct(r, leg):
    d = (r.get("legs") or {}).get(leg)
    return None if not d else d.get("pct_vs_ref")


# Figure 1 is now make_icecost_fig.py: the ice cost, not the model step, and without the delayed
# exchange, which Sect. 6.6 rules out. The old model-step version led with that scheme's bars and
# would contradict the report if it were rebuilt.

# NOTE (2026-08-06): the report's figures are now made by make_icecost_fig.py (Fig. 1, the ice
# cost) and make_artefact_fig.py (Fig. 2, the decomposition imprint). The scaling and accuracy
# figures that used to live here described the delayed exchange, which the report rejects; they
# were removed with the sections that carried them rather than left to be rebuilt by accident.
