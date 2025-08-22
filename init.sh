#!/bin/bash

# init venv and install requirements
python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt

# download and resplit UD data
bash preprocess.sh

# run example training in mono and multi setting, with an extremely small transformer for 1 epoch
bash example-run.sh