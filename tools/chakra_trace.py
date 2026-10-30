"""Decode, check and compare Chakra execution traces written by `setep --chakra`.

Uses only the Python standard library: the protobuf wire format is decoded
directly, so neither Chakra nor ASTRA-sim needs to be installed.

  python chakra_trace.py summary PREFIX           counts per rank
  python chakra_trace.py check PREFIX             structural checks (exit 1 on failure)
  python chakra_trace.py compare PREFIX_A PREFIX_B [OLD=NEW ...]
      decoded node-by-node comparison of two exports: node IDs, names, types,
      dependencies, durations, tensor shapes and every attribute. OLD=NEW
      treats attribute name OLD as NEW, for traces written by other tools.

PREFIX is the path given by the manifest without extension, e.g. out/chatgpt2
for out/chatgpt2.chakra.json and out/chatgpt2.<rank>.et.
"""
import json
import sys
from pathlib import Path

COMPUTE, SEND, RECV = 4, 5, 6
RENAMED = {}  # attribute aliases for compare, OLD -> NEW


def varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        assert shift < 70, 'invalid protobuf varint'


def fields(data):
    result = {}
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            assert len(value) == size
            offset += size
        else:
            raise AssertionError(f'unexpected protobuf wire type {wire}')
        result.setdefault(number, []).append(value)
    return result


def records(path):
    data = Path(path).read_bytes()
    offset = 0
    result = []
    while offset < len(data):
        size, offset = varint(data, offset)
        assert offset + size <= len(data)
        result.append(fields(data[offset:offset + size]))
        offset += size
    return result


def io_info(encoded):
    if not encoded:
        return None
    info = fields(encoded[0])
    return tuple(info.get(n, [b''])[0].decode() for n in (2, 3))


def decode(path):
    """Nodes of one rank file, in file order. Asserts unique IDs and local,
    earlier dependencies (receives stand in for remote producers)."""
    metadata, *nodes = records(path)
    assert metadata[1] == [b'0.0.4'], 'unexpected Chakra schema version'
    result, seen = [], set()
    for node in nodes:
        identifier = node[1][0]
        assert identifier not in seen, (path, 'duplicate node id', identifier)
        attributes = {}
        for encoded in node.get(10, []):
            attribute = fields(encoded)
            name = attribute[1][0].decode()
            values = [(field, values[0]) for field, values in attribute.items() if field != 1]
            assert len(values) == 1
            attributes[RENAMED.get(name, name)] = values[0]
        deps = node.get(5, [])
        assert set(deps) <= seen, (path, identifier, 'dependency not earlier in the same file')
        seen.add(identifier)
        result.append({'id': identifier, 'name': node[2][0].decode(), 'type': node[3][0], 'deps': deps,
                       'duration': node.get(7, [0])[0], 'inputs': io_info(node.get(8)),
                       'outputs': io_info(node.get(9)), 'attrs': attributes,
                       'fields': {k: len(v) for k, v in node.items()}})
    return result


def load(prefix):
    prefix = Path(prefix)
    manifest = json.loads(Path(str(prefix) + '.chakra.json').read_text())
    ranks = [decode(f'{prefix}.{rank}.et') for rank in range(manifest['num_ranks'])]
    extra = [p for p in prefix.parent.glob(prefix.name + '.*.et')
             if not p.name[len(prefix.name) + 1:-3].isdigit() or int(p.name[len(prefix.name) + 1:-3]) >= len(ranks)]
    assert not extra, ('rank files beyond the manifest', extra)
    return manifest, ranks


def check(prefix):
    """Every send has one matching receive of equal size, compute counts agree
    with the manifest, and receives carry no cross-file dependencies."""
    manifest, ranks = load(prefix)
    sends, recvs = {}, {}
    compute = edges = 0
    for rank, nodes in enumerate(ranks):
        for node in nodes:
            attrs = node['attrs']
            edges += len(node['deps'])
            if node['type'] == COMPUTE:
                compute += 1
                continue
            assert node['type'] in (SEND, RECV), ('unexpected node type', node['type'])
            key = tuple(attrs[k][1] for k in ('comm_src', 'comm_dst', 'comm_tag'))
            assert key[0] != key[1], key
            assert rank == key[0 if node['type'] == SEND else 1], (rank, key)
            collection = sends if node['type'] == SEND else recvs
            assert key not in collection, ('duplicate transfer', key)
            collection[key] = (attrs['comm_size'][1], node)
            if node['type'] == SEND:
                assert len(node['deps']) == 1, ('send must depend on its producer', key)
            else:
                assert not node['deps'], ('receive with dependencies', key)
    assert sends.keys() == recvs.keys(), 'unmatched sends or receives'
    assert all(sends[k][0] == recvs[k][0] for k in sends), 'send/receive size mismatch'
    assert compute == manifest['compute_instances'], (compute, manifest['compute_instances'])
    assert len(sends) == manifest['transfers'], (len(sends), manifest['transfers'])
    for rank in manifest['ranks']:
        assert rank['resident_mb'] <= rank['capacity_mb'] + 1e-9, rank
    return dict(num_ranks=manifest['num_ranks'], compute_nodes=compute, transfers=len(sends),
                comm_bytes=sum(s for s, _ in sends.values()), dependency_edges=edges,
                rank_pairs=len({k[:2] for k in sends}))


def compare(a, b):
    """Return a list of differences between two exports (empty when equal)."""
    (ma, ra), (mb, rb) = load(a), load(b)
    differences = []
    for key in sorted(set(ma) | set(mb)):
        if key == 'workload_prefix':
            continue
        if ma.get(key) != mb.get(key):
            differences.append(f'manifest {key}: {ma.get(key)!r} != {mb.get(key)!r}')
    for rank, (na, nb) in enumerate(zip(ra, rb)):
        if len(na) != len(nb):
            differences.append(f'rank {rank}: {len(na)} nodes != {len(nb)} nodes')
        for x, y in zip(na, nb):
            if x != y:
                differences.append(f'rank {rank} node {x["id"]}: {x} != {y}')
                break
    return differences


def main(argv):
    if len(argv) == 3 and argv[1] == 'summary':
        manifest, ranks = load(argv[2])
        for rank, nodes in enumerate(ranks):
            kinds = [n['type'] for n in nodes]
            print(f'rank {rank}: {kinds.count(COMPUTE)} compute, {kinds.count(SEND)} send, {kinds.count(RECV)} recv')
        return 0
    if len(argv) == 3 and argv[1] == 'check':
        print(json.dumps(check(argv[2])))
        return 0
    if len(argv) >= 4 and argv[1] == 'compare':
        RENAMED.update(alias.split('=', 1) for alias in argv[4:])
        differences = compare(argv[2], argv[3])
        for line in differences[:20]:
            print(line)
        nodes = sum(len(r) for r in load(argv[2])[1])
        print('IDENTICAL' if not differences else f'{len(differences)} DIFFERENCES', f'({nodes} nodes)')
        return 1 if differences else 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
