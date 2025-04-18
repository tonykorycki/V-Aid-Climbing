#!/bin/bash
source .venv/bin/activate
xvfb-run -a python src/pi_CV_main.py "$@"
