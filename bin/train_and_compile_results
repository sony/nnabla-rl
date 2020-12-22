#!/bin/bash
if [ $# -ne 2 ]; then
    echo "usage: $0 <script_file_name> <env>"
    exit 1
fi
RESULTDIR="./$2_results"
for seed in 1 10 100
do
    python $1 --seed $seed --env $2
done
cd $RESULTDIR
compile_results --outdir compile_results