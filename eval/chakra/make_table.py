"""Collect every DIR/*/sweep_*.json into DIR/chakra_table.txt.

Usage: python make_table.py [DIR]   (default: this directory)
"""
import glob, json, os, sys
here = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(here, '*', 'sweep_*.json'))):
    if os.sep + 'probe' + os.sep in f:
        continue
    rows += json.load(open(f))
cols = ['model', 'memMbPerCube', 'replicas', 'num_ranks', 'compute_nodes', 'send_nodes',
        'recv_nodes', 'matched_pairs', 'rank_pairs', 'comm_bytes', 'trace_bytes',
        'resident_mb_sum', 'gen_seconds', 'check']
order = {'gpt2': 0, 'switch': 1, 'llama': 2, 'dsr1': 3}
rows.sort(key=lambda r: (order.get(r['model'], 9), r.get('replicas', 1), -r['memMbPerCube']))
for r in rows:
    r.setdefault('replicas', 1)
w = [max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols]
out = ['  '.join(c.rjust(x) for c, x in zip(cols, w))]
out += ['  '.join(str(r.get(c, '')).rjust(x) for c, x in zip(cols, w)) for r in rows]
open(os.path.join(here, 'chakra_table.txt'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
