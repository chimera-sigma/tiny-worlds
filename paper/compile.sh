#!/bin/bash
# Compile the paper with pdflatex

set -e

echo "[COMPILE] Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode paper.tex

echo "[COMPILE] Running pdflatex (second pass for references)..."
pdflatex -interaction=nonstopmode paper.tex

echo "[COMPILE] Cleaning up auxiliary files..."
rm -f paper.aux paper.log paper.out

echo "[COMPILE] Done! Output: paper.pdf"
ls -lh paper.pdf
