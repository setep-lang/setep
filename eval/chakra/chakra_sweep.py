"""Result set 3: distributed Chakra export sweep + independent .et decoding.

Usage: python chakra_sweep.py SETEP_ROOT OUT_DIR LABEL=models/x/y.setep[,LABEL=...] MEM1,MEM2,... [replicas=N]

For each model and memory capacity (MB per processor): run `setep --chakra`
into its own case directory, time it, then decode every .et file with
tools/chakra_trace.py and check send/receive matching.

Paper settings (the inventory used for every published case): 64 nodes x 64
accelerators (4096, no host processors), one replica unless replicas=N is
given, one sequence.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
models = dict(item.split('=', 1) for item in sys.argv[3].split(','))
mems = [int(m) for m in sys.argv[4].split(',')]
extra = {k: int(v) for k, v in (kv.split('=') for kv in sys.argv[5].split(','))} if len(sys.argv) > 5 else {}
setep = root / 'build' / ('setep.exe' if os.name == 'nt' else 'setep')

spec = importlib.util.spec_from_file_location('chakra_trace', root / 'tools' / 'chakra_trace.py')
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

INVENTORY = ['--nodes', '64', '--processors-per-node', '64', '--hosts-per-node', '0', '--sequences', '1']


def analyse(case, stem):
    prefix = case / stem
    manifest, ranks = ct.load(prefix)
    summary = ct.check(prefix)  # asserts matched sends/receives, local deps, counts
    cpu_compute = sum(int(n['attrs'].get('is_cpu_op', (0, 0))[1]) for r in ranks for n in r if n['type'] == ct.COMPUTE)
    return dict(num_ranks=manifest['num_ranks'], manifest_transfers=manifest['transfers'],
                manifest_compute_instances=manifest['compute_instances'],
                compute_nodes=summary['compute_nodes'], cpu_compute_nodes=cpu_compute,
                send_nodes=summary['transfers'], recv_nodes=summary['transfers'],
                matched_pairs=summary['transfers'], comm_bytes=summary['comm_bytes'],
                rank_pairs=summary['rank_pairs'], dep_edges=summary['dependency_edges'],
                trace_bytes=sum((case / f'{stem}.{r}.et').stat().st_size for r in range(manifest['num_ranks'])),
                manifest_bytes=(case / f'{stem}.chakra.json').stat().st_size,
                resident_mb_max=max(r.get('resident_mb', 0) for r in manifest['ranks']),
                resident_mb_sum=round(sum(r.get('resident_mb', 0) for r in manifest['ranks']), 1),
                capacity_mb=manifest['ranks'][0].get('capacity_mb'))


rows = []
tag = ''.join(f'_{k}{v}' for k, v in extra.items())
for label, model in models.items():
    for mem in mems:
        case = out / f'{label}_mem{mem}{tag}'
        shutil.rmtree(case, ignore_errors=True)
        case.mkdir(parents=True)
        stem = Path(model).stem
        command = [str(setep), str(root / model), '--chakra', '-o', str(case), *INVENTORY,
                   '--memory-mb', str(mem), '--replicas', str(extra.get('replicas', 1))]
        (case / 'command.txt').write_text(' '.join(command) + '\n')
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        wall = time.perf_counter() - t0
        (case / 'run.log').write_text(proc.stdout + proc.stderr)
        row = dict(model=label, memMbPerCube=mem, **extra, exit=proc.returncode, gen_seconds=round(wall, 3))
        if proc.returncode == 0 and (case / f'{stem}.chakra.json').exists():
            try:
                row.update(analyse(case, stem))
                row['check'] = 'OK' if row['send_nodes'] == row['manifest_transfers'] else 'FAIL'
            except AssertionError as e:
                row['check'] = f'ASSERT {e!r}'[:300]
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            row['check'] = 'RUN FAILED: ' + ' | '.join(tail)[:300]
        print(json.dumps(row), flush=True)
        rows.append(row)
json.dump(rows, open(out / f'sweep_{"_".join(models)}{tag}.json', 'w'), indent=1)
