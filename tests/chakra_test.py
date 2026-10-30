"""Chakra export: structural checks of decoded traces (matched sends and
receives, local dependencies, manifest counts, memory capacity), placement
behaviour, and the GPT-2 / Switch numbers reported with the paper settings."""
import json

from common import ROOT, arguments, setep, source
import chakra_trace

args = arguments()
INVENTORY = ['--nodes', '64', '--processors-per-node', '64']


def export(model, out, *options):
    setep(args, model, '--chakra', '-o', out, *INVENTORY, *options)
    prefix = out / model.stem
    return json.loads(prefix.with_suffix('.chakra.json').read_text()), chakra_trace.check(prefix), prefix


# A three-stage chain split across processors by memory capacity.
chain = source(args, 'chain', 'A=2%4@fp32 w=4%4@fp32 v=4%4@fp32 u=4%4@fp32 '
                              'RUN(A){A*w >> B; B*v >> C; C*u >> D}')
manifest, summary, prefix = export(chain, args.work / 'chain', '--memory-mb', '0.0002')
assert manifest['num_ranks'] == 3 and summary['transfers'] == 2, (manifest, summary)
assert summary['comm_bytes'] == 2 * 32
nodes = chakra_trace.decode(f'{prefix}.1.et')
assert [n['type'] for n in nodes] == [chakra_trace.RECV, chakra_trace.COMPUTE, chakra_trace.SEND]
compute = nodes[1]
assert compute['name'] == 'MATMUL' and compute['deps'] == [nodes[0]['id']]
assert compute['attrs']['num_ops'] == (13, 64) and compute['attrs']['tensor_size'] == (13, 96)
assert compute['inputs'] == ('[[2,4]]', '[fp32]') and compute['outputs'] == ('[[2,4]]', '[fp32]')
assert compute['attrs']['setep_operation_id'] == (9, 1)
# Everything fits on one processor: no communication.
manifest, summary, _ = export(chain, args.work / 'chain_one', '--memory-mb', '1')
assert manifest['num_ranks'] == 1 and summary['transfers'] == 0 and summary['compute_nodes'] == 3
# Replicas and sequences multiply the compute instances; a smaller export
# removes the stale rank files of a larger one.
manifest, summary, prefix = export(chain, args.work / 'chain', '--memory-mb', '1', '--replicas', '3', '--sequences', '2')
assert manifest['num_ranks'] == 3 and manifest['replicas'] == 3 and manifest['sequences_per_replica'] == 2
assert summary['compute_nodes'] == 18 and summary['transfers'] == 0
manifest, summary, _ = export(chain, args.work / 'chain', '--memory-mb', '1')
assert manifest['num_ranks'] == 1 and not (args.work / 'chain' / 'chain.1.et').exists()
# Placement failures are reported.
code, output = setep(args, chain, '--chakra', '-o', args.work / 'fail', '--memory-mb', '0.00001', check=False)
assert code == 1 and 'exceeds the per-processor memory capacity' in output, output
code, output = setep(args, chain, '--chakra', '-o', args.work / 'fail', '--nodes', '1', '--processors-per-node', '2',
                     '--memory-mb', '0.0002', check=False)
assert code == 1 and 'do not fit' in output, output
# Host processors are part of the inventory but never receive operations.
manifest, summary, _ = export(chain, args.work / 'hosts', '--memory-mb', '0.0002', '--hosts-per-node', '2')
assert all(rank['processor'] == 'gpu' for rank in manifest['ranks'])

# Paper settings: one replica, one sequence, 64 x 64 inventory.
for model, memory, expected in [
        ('chatgpt2/chatgpt2.setep', 500, dict(num_ranks=12, compute_nodes=7468, transfers=285, comm_bytes=10940204)),
        ('chatgpt2/chatgpt2.setep', 100000, dict(num_ranks=1, compute_nodes=7468, transfers=0, comm_bytes=0)),
        ('switch_base_8/switch_base_8.setep', 2000, dict(num_ranks=41, compute_nodes=74157, transfers=1180, comm_bytes=45320675))]:
    _, summary, _ = export(ROOT / 'models' / model, args.work / f'paper_{memory}', '--memory-mb', str(memory),
                           '--replicas', '1', '--sequences', '1')
    for key, value in expected.items():
        assert summary[key] == value, (model, memory, key, summary[key], value)
print('Chakra traces are consistent; GPT-2 and Switch match the paper counts')
