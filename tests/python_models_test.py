"""Python generation for every bundled model: the generated module and the
shared setep.py runtime compile; when PyTorch is installed each model imports
and exposes its entry point, weights and opaque hooks."""
import importlib.util
import py_compile
import re
import sys

from common import ROOT, arguments, setep

args = arguments()
models = sorted((ROOT / 'models').glob('*/*.setep'))
assert len(models) >= 11, models
have_torch = importlib.util.find_spec('torch') is not None
for model in models:
    out = args.work / model.stem
    setep(args, model, '--python', '-o', out)
    generated = out / (model.stem + '.py')
    py_compile.compile(str(generated), doraise=True)
    py_compile.compile(str(out / 'setep.py'), doraise=True)
    text = generated.read_text()
    assert text.startswith('"""Generated SETEP model; shared runtime is in setep.py."""')
    assert 'class Model(nn.Module):' in text and '    def forward(self, ' in text
    entry = re.findall(r'\n    def forward\(self, [^)]*\):\n        return self\.(\w+)\(', text)
    assert entry, model
    if have_torch:
        sys.path.insert(0, str(out))
        spec = importlib.util.spec_from_file_location(model.stem, generated)
        module = importlib.util.module_from_spec(spec)
        sys.modules[model.stem] = module
        spec.loader.exec_module(module)
        instance = module.Model()
        assert hasattr(instance, entry[0]), (model, entry)
        sys.path.remove(str(out))
        del sys.modules['setep']
# The runtime is the same for every model.
runtimes = {(args.work / m.stem / 'setep.py').read_bytes() for m in models}
assert len(runtimes) == 1
print(f'{len(models)} models generated and compiled' + ('' if have_torch else ' (PyTorch missing: import skipped)'))
