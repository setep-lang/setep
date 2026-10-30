"""Shared helpers for the SETEP ctest suite (standard library only)."""
import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP = 77  # ctest SKIP_RETURN_CODE


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setep', type=pathlib.Path, required=True, help='setep executable')
    parser.add_argument('--work', type=pathlib.Path, required=True, help='scratch directory')
    args = parser.parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)
    return args


def setep(args, *options, check=True):
    """Run the setep tool; returns (exit code, stdout + stderr)."""
    result = subprocess.run([str(args.setep), *map(str, options)], capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result.returncode, result.stdout + result.stderr


def source(args, name, text):
    path = args.work / (name + '.setep')
    path.write_text(text)
    return path


def require_modules(*names):
    """Exit with the ctest skip code when an optional package is missing."""
    import importlib.util
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        print('skipped: missing Python packages ' + ', '.join(missing) + ' in ' + sys.executable)
        raise SystemExit(SKIP)


sys.path.insert(0, str(ROOT / 'tools'))
