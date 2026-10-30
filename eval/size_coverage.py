"""Result set 1: SETEP description size and coverage.

Usage: python size_coverage.py SETEP_ROOT OUT_DIR
Copies every models/*/*.setep into OUT_DIR/gen/<model>/, runs `setep --python`
on the copy, and counts source/generated lines.  Writes size_coverage.csv/.txt.
OUT_DIR/gen/chatgpt2 is the GEN_DIR used by gpt2/compare_hf.py and onnx_gpt2.py.
"""
import csv, glob, os, re, shutil, subprocess, sys, time

repo, out = sys.argv[1], sys.argv[2]
gen_exe = os.path.join(repo, 'build', 'setep.exe' if os.name == 'nt' else 'setep')
IDENT = r'[A-Za-z_][A-Za-z_0-9]*'


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return '\n'.join(line.split('//', 1)[0] for line in text.splitlines())


def top_level_definitions(code):
    """Yield (name, params, body) for NAME(params) { body } at brace depth 0."""
    i, depth, n = 0, 0, len(code)
    head = re.compile(r'(' + IDENT + r')\s*\(([^()]*)\)\s*\{')
    while i < n:
        if depth == 0:
            m = head.match(code, i)
            if m and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] in '_>')):
                j, d = m.end(), 1
                while d:
                    d += {'{': 1, '}': -1}.get(code[j], 0)
                    j += 1
                yield m.group(1), m.group(2), code[m.end():j - 1]
                i = j
                continue
        c = code[i]
        depth += {'{': 1, '}': -1}.get(c, 0)
        i += 1


def classify(body):
    stripped = body.strip()
    if not stripped:
        return 'abstract'
    # Body made only of cost annotations: key = number[, ...]
    parts = [p.strip() for p in re.split(r'[;,\n]', stripped) if p.strip()]
    if all(re.fullmatch(r'[a-z_][a-z_0-9]*\s*=\s*[-+0-9.eE]+', p) for p in parts):
        return 'abstract_annotated'
    return 'concrete'


def count_lines(path):
    if not os.path.exists(path):
        return None, None
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    return len(lines), sum(1 for l in lines if l.strip())


rows = []
for src in sorted(glob.glob(os.path.join(repo, 'models', '*', '*.setep'))):
    model = os.path.splitext(os.path.basename(src))[0]
    text = open(src, encoding='utf-8').read()
    code = strip_comments(text)
    sloc = sum(1 for l in code.splitlines() if l.strip())
    defs = list(top_level_definitions(code))
    kinds = [classify(b) for _, _, b in defs]
    repeats = re.findall(IDENT + r'\s*/\s*[A-Za-z_0-9]+\s*(?:\[[^\]]*\])?\s*\{', code)
    weights = re.findall(r'(?m)^\s*([a-z_][a-z_0-9]*)\s*(\[[^\]]*\])?\s*=\s*[^;\n]*@\w+', code)
    weight_arrays = sum(1 for _, idx in weights if idx)
    work = os.path.join(out, 'gen', model)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    copy = os.path.join(work, model + '.setep')
    shutil.copy(src, copy)
    t0 = time.perf_counter()
    proc = subprocess.run([gen_exe, copy, '--python'], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    py_total, py_nb = count_lines(os.path.join(work, model + '.py'))
    rt_total, _ = count_lines(os.path.join(work, 'setep.py'))
    err = (proc.stdout + proc.stderr).strip().replace('\n', ' | ')
    rows.append(dict(
        file=os.path.relpath(src, repo).replace('\\', '/'),
        setep_total_lines=text.count('\n') + (0 if text.endswith('\n') else 1),
        setep_sloc=sloc,
        computations=len(defs),
        concrete=kinds.count('concrete'),
        abstract_empty=kinds.count('abstract'),
        abstract_cost_annotated=kinds.count('abstract_annotated'),
        repeated_scopes=len(repeats),
        weight_decls=len(weights),
        weight_array_decls=weight_arrays,
        gen_exit=proc.returncode,
        gen_seconds=round(dt, 3),
        py_lines=py_total, py_nonblank=py_nb,
        runtime_setep_py_lines=rt_total,
        abstract_names=' '.join(n for (n, _, _), k in zip(defs, kinds) if k != 'concrete'),
        gen_output='' if proc.returncode == 0 else err[:400],
    ))

keys = list(rows[0])
with open(os.path.join(out, 'size_coverage.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)
cols = ['file', 'setep_sloc', 'computations', 'concrete', 'abstract_empty',
        'abstract_cost_annotated', 'repeated_scopes', 'weight_decls',
        'weight_array_decls', 'gen_exit', 'py_lines']
widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
lines = ['  '.join(c.ljust(w) for c, w in zip(cols, widths))]
for r in rows:
    lines.append('  '.join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
table = '\n'.join(lines)
open(os.path.join(out, 'size_coverage.txt'), 'w').write(table + '\n')
print(table)
for r in rows:
    if r['gen_exit'] != 0:
        print('FAIL', r['file'], r['gen_output'])
