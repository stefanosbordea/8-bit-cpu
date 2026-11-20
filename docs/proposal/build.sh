#!/bin/sh
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode proposal.tex >/dev/null
xelatex -interaction=nonstopmode proposal.tex >/dev/null
echo "proposal.pdf rebuilt"
