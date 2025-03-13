#!/bin/bash

pwd

ls
LB_ROOT=/root/open_asr_leaderboard/
script_name=$(basename $1)
deps=$(basename $(dirname $1))

# common dependencies
# Should be already installed if requirements.txt in 
# /docker and in /requirements are maintained consistent
pip install -r $LB_ROOT/requirements/requirements.txt

# library specific dependencies
pip install -r $LB_ROOT/requirements/requirements_${deps}.txt

[ -f ~/.cache/huggingface/token ] || huggingface-cli login

pushd $LB_ROOT/$deps/

bash $LB_ROOT/$deps/$script_name

popd