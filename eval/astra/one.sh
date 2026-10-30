#!/bin/bash
# Run ASTRA-sim on one exported case. usage: one.sh TRACE_PREFIX
#   e.g. one.sh .../eval/chakra/gpt2/gpt2_mem500/chatgpt2
# The network has one Ring endpoint per rank in the manifest.
set -u
PREFIX=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ASTRA=${ASTRA_SIM:-~/work/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware}
ranks=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['num_ranks'])" $PREFIX.chakra.json)
net=$(mktemp --suffix=.yml)
printf 'topology: [ Ring ]\nnpus_count: [ %s ]\nbandwidth: [ 50.0 ]\nlatency: [ 500.0 ]\n' $ranks > $net
time $ASTRA --workload-configuration=$PREFIX \
  --system-configuration=$ROOT/models/chatgpt2/chatgpt2_chunk.json \
  --network-configuration=$net \
  --remote-memory-configuration=$HERE/no_memory_expansion.json > /tmp/one.log 2>&1
echo exit=$?; tail -15 /tmp/one.log; rm -f $net
