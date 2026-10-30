#!/bin/bash
# Run ASTRA-sim (analytical, congestion unaware) on every exported trace case
# under EVAL_DIR/chakra/<model>/<case>/ (written by eval/chakra/chakra_sweep.py).
# usage: run_astra.sh EVAL_DIR
#   ASTRA_SIM overrides the simulator binary; the default is the analytical
#   build of a checkout at ~/work/astra-sim.
# On Windows the traces can be read from WSL, e.g.
#   MSYS_NO_PATHCONV=1 wsl bash /mnt/c/.../setep/eval/astra/run_astra.sh /mnt/c/.../setep/eval
set -u
EVAL=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ASTRA=${ASTRA_SIM:-~/work/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware}
SYSTEM=$ROOT/models/chatgpt2/chatgpt2_chunk.json
MEMORY=$HERE/no_memory_expansion.json
OUT=$EVAL/astra; mkdir -p $OUT
echo "case,ranks,exit,finished_ranks,max_finish_cycles,sum_finish_cycles,sum_gpu_cycles,max_exposed_comm_cycles,wall_s" > $OUT/results.csv
for manifest in $(ls $EVAL/chakra/*/*/*.chakra.json | sort); do
  dir=$(dirname $manifest); case=$(basename $dir); prefix=${manifest%.chakra.json}
  ranks=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['num_ranks'])" $manifest)
  net=$OUT/net_$ranks.yml
  printf 'topology: [ Ring ]\nnpus_count: [ %s ]\nbandwidth: [ 50.0 ]\nlatency: [ 500.0 ]\n' $ranks > $net
  log=$OUT/$case.log; start=$(date +%s.%N)
  timeout 1800 $ASTRA --workload-configuration=$prefix \
    --system-configuration=$SYSTEM \
    --network-configuration=$net \
    --remote-memory-configuration=$MEMORY > $log 2>&1
  code=$?; wall=$(python3 -c "import time;print(round(time.time()-$start,2))")
  stats=$(grep -oE "sys\[[0-9]+\] finished, [0-9]+ cycles" $log | awk '{c=$3+0; n++; s+=c; if(c>m)m=c} END{printf "%d,%d,%d", n, m, s}')
  gpu=$(grep -oE "GPU time: [0-9]+" $log | awk '{s+=$3} END{printf "%d", s}')
  comm=$(grep -oE "exposed communication [0-9]+" $log | awk '{c=$3+0; if(c>m)m=c} END{printf "%d", m}')
  echo "$case,$ranks,$code,$stats,$gpu,$comm,$wall" | tee -a $OUT/results.csv
done
