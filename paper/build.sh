#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Building TopoPRM paper ==="

echo "[1/4] pdflatex (first pass)..."
pdflatex -interaction=nonstopmode main.tex

echo "[2/4] bibtex..."
bibtex main

echo "[3/4] pdflatex (second pass)..."
pdflatex -interaction=nonstopmode main.tex

echo "[4/4] pdflatex (third pass)..."
pdflatex -interaction=nonstopmode main.tex

echo "=== Build complete: main.pdf ==="
