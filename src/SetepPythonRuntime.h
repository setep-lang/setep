#pragma once

// Text of the shared setep.py support module imported by every generated model.
namespace setep {
inline constexpr const char* pythonRuntime=R"PY("""Generated from SETEP semantic IR. Opaque operations are demo placeholders."""
from dataclasses import dataclass
import contextlib
from fractions import Fraction
import math
import os
import torch
from torch import Tensor, nn

def normalize(value, eps=1e-5):
    """Normalize the final dimension, without learned affine parameters."""
    return torch.nn.functional.layer_norm(value, (value.shape[-1],), eps=eps)

def causal_softmax(scores, mask):
    """Scaled attention over valid keys, including cached decode offsets."""
    query_count, key_count = scores.shape[-2:]
    allowed = torch.ones(query_count, key_count, dtype=torch.bool,
                         device=scores.device).tril(key_count - query_count)
    allowed = allowed & mask.to(device=scores.device, dtype=torch.bool)
    masked = scores.masked_fill(~allowed, float('-inf'))
    valid = allowed.any(dim=-1, keepdim=True)
    return torch.softmax(masked.masked_fill(~valid, 0), dim=-1).masked_fill(~valid, 0)

def cross_entropy(logits, targets, ignore_index=-100):
    """Mean next-token loss; padding targets must use ignore_index."""
    if not bool((targets != ignore_index).any()):
        raise ValueError('cross entropy requires at least one target token')
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1).long(), ignore_index=ignore_index).reshape(1)

def tensor_gradient(output, value, upstream=None, *, retain_graph=True):
    """Vector-Jacobian product for an explicit SETEP backward computation."""
    if upstream is None:
        if output.numel() != 1:
            raise ValueError('non-scalar gradients require an upstream tensor')
        upstream = torch.ones_like(output)
    return torch.autograd.grad(output, value, upstream,
                               retain_graph=retain_graph)[0]

def local_gradients(function, inputs, upstream):
    """Recompute a local operation and return all of its vector-Jacobian products."""
    with torch.enable_grad():
        leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
        output = function(*leaves)
        return tuple(value.detach() for value in torch.autograd.grad(output, leaves,
            upstream.detach(), retain_graph=False))

def mlp(value, up, up_bias, down, down_bias):
    return torch.nn.functional.gelu(value @ up + up_bias, approximate='tanh') @ down + down_bias

def attention(value, mask, q, k, v, o, qb, kb, vb, ob, heads):
    width = value.shape[-1] // heads
    query, key, content = [(value @ weight + bias).reshape(-1, heads, width).transpose(0, 1)
                          for weight, bias in ((q, qb), (k, kb), (v, vb))]
    probabilities = causal_softmax((query @ key.transpose(-2, -1)) / width ** .5, mask)
    result = (probabilities @ content).transpose(0, 1).reshape(value.shape)
    return result @ o + ob

def sgd_update(parameter, gradient, learning_rate):
    """Update an existing parameter in place, preserving weight sharing."""
    if parameter.shape != gradient.shape:
        raise ValueError('SGD parameter and gradient shapes must match')
    with torch.no_grad():
        parameter.add_(gradient, alpha=-float(learning_rate))
    return parameter

def child_path(instance_path, *indices):
    return (*instance_path, *indices)

_log_index = 0
_log_enabled = False

def set_debug(enabled=True):
    global _log_enabled
    _log_enabled = bool(enabled)

def _log(message, *, always=False):
    global _log_index
    if not (_log_enabled or always):
        return
    import textwrap
    message = str(message)
    # Reserve enough prefix space even when wrapping crosses an index digit.
    width = 80 - len(f'[SETEP {_log_index + len(message) + 1}] ')
    for line in textwrap.wrap(message, width=width):
        _log_index += 1
        print(f'[SETEP {_log_index}] {line}', flush=True)

def _binary_shapes(operation, left, right, result):
    if not isinstance(result, _Shape):
        _log(f'{operation} {tuple(left.shape)} {tuple(right.shape)}'
             f' -> {tuple(result.shape)}')
    return result

def tensor_op(op, args):
    a = args[0]
    if isinstance(a, _Shape):
        if op in ('sum_last', 'mean_last', 'max_last'):
            one = a.plan.dimension()
            a.plan.equal({one: 1, -1: -1})
            result = _Shape(a.plan, one if a.rank == 1 else a.rows, one)
            result.rank = a.rank
            result.singleton_rows = a.rank == 1 or a.singleton_rows
            result.singleton_cols = True
            return result
        if op in ('mul', 'div', 'maximum'):
            return a + args[1]
        return a
    if op == 'sum_last': return a.sum(dim=-1, keepdim=True)
    if op == 'mean_last': return a.mean(dim=-1, keepdim=True)
    if op == 'max_last': return a.max(dim=-1, keepdim=True).values
    if op == 'softmax_last': return torch.softmax(a, dim=-1)
    if op == 'layernorm_last': return torch.nn.functional.layer_norm(a, (a.shape[-1],), eps=1e-5)
    if op == 'rmsnorm_last': return a * torch.rsqrt(a.square().mean(dim=-1, keepdim=True) + 1e-5)
    if op == 'mul': return a * args[1]
    if op == 'div': return a / args[1]
    if op == 'maximum': return torch.where((a >= args[1]) | torch.isnan(a), a, args[1])
    if op == 'exp': return torch.exp(a)
    if op == 'rsqrt_eps': return torch.rsqrt(a + 1e-5)
    if op.startswith('divide_by:'): return a / float(op.split(':')[1])
    raise ValueError(f'unknown tensor operation: {op}')

def _matmul(left, right):
    return _binary_shapes('matmul', left, right, left @ right)

def _add(left, right):
    return _binary_shapes('add', left, right, left + right)

def split(value, count, index):
    if isinstance(value, _Shape):
        width = value.plan.dimension()
        value.plan.equal({value.rows if value.rank == 1 else value.cols: 1, width: -count})
        result = _Shape(value.plan, width if value.rank == 1 else value.rows,
                        value.cols if value.rank == 1 else width)
        result.rank = value.rank
        result.singleton_rows = value.singleton_rows if value.rank == 2 else False
        result.singleton_cols = value.rank == 1
        return result
    if count <= 0 or not 0 <= index < count or value.shape[-1] % count:
        raise ValueError("invalid last-axis split")
    return value.narrow(-1, index * (value.shape[-1] // count), value.shape[-1] // count)

def merge(values):
    if not values:
        raise ValueError("cannot merge empty family")
    if isinstance(values[0], _Shape):
        first = values[0]
        width = first.plan.dimension()
        equation = {width: -1}
        for value in values:
            first.plan.match(first.cols if first.rank == 1 else first.rows,
                             value.cols if first.rank == 1 else value.rows)
            dimension = value.rows if first.rank == 1 else value.cols
            equation[dimension] = equation.get(dimension, 0) + 1
        first.plan.equal(equation)
        result = _Shape(first.plan, width if first.rank == 1 else first.rows,
                        first.cols if first.rank == 1 else width)
        result.rank = first.rank
        result.singleton_rows = first.singleton_rows if first.rank == 2 else False
        result.singleton_cols = first.rank == 1
        return result
    return torch.cat(values, dim=-1)

def require_shape(value, dims):
    if isinstance(value, _Shape):
        rows, cols = (1, dims[0]) if len(dims) == 1 else dims
        value.plan.equal({value.rows: 1, -1: -rows})
        value.plan.equal({value.cols: 1, -1: -cols})
    elif list(value.shape) != dims:
        raise ValueError('declared tensor shape mismatch')
    return value

def split_axis(value, count, index, axis):
    if isinstance(value, _Shape):
        if value.rank == 1 and axis in (0, -1): return split(value, count, index)
        if axis == 0:
            return split(value.transpose(-2, -1), count, index).transpose(-2, -1)
        if axis in (1, -1):
            return split(value, count, index)
        raise ValueError('shape inference supports two axes')
    if count <= 0 or not 0 <= index < count or value.shape[axis] % count:
        raise ValueError('invalid split count')
    size = value.shape[axis] // count
    return value.narrow(axis, index * size, size)

def merge_axis(values, axis):
    if isinstance(values[0], _Shape):
        if axis == 0:
            return merge([v.transpose(-2, -1) for v in values]).transpose(-2, -1)
        if axis in (1, -1):
            return merge(values)
        raise ValueError('shape inference supports two axes')
    return torch.cat(values, dim=axis)

def require_array_size(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("array count mismatch")
    return values

def _same_shape(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return isinstance(left, list) and isinstance(right, list) and len(left) == len(right) and all(_same_shape(a, b) for a, b in zip(left, right))
    if isinstance(left, _Shape) and isinstance(right, _Shape):
        left.plan.match(left.rows, right.rows)
        left.plan.match(left.cols, right.cols)
        return True
    return left.shape == right.shape

def array_push(values, value):
    if values and not _same_shape(values[0], value):
        raise ValueError('array elements must have identical shapes')
    values.append(value)

def checked_array(values):
    if values and not all(_same_shape(values[0], value) for value in values):
        raise ValueError('array elements must have identical shapes')
    return values

def merge_group(values):
    return merge(values)

def reduce_sum(values):
    if not values:
        raise ValueError("cannot reduce empty family")
    result = values[0]
    for value in values[1:]:
        if not isinstance(result, _Shape) and result.shape != value.shape:
            raise ValueError('reduction shape mismatch')
        result = _add(result, value)
    return result

def reduce_scatter(values):
    if not values or not isinstance(values[0], list):
        raise ValueError('reduce scatter requires an array of arrays')
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError('reduce scatter array size mismatch')
    return [reduce_sum([row[i] for row in values]) for i in range(width)]

def group_member(values, count, index):
    if len(values) != count:
        raise ValueError('corresponding transfer cardinality mismatch')
    return values[index]

def selected(selector, count, reverse):
    if isinstance(selector, _Shape):
        if selector.occurrence is None:
            raise ValueError("demo selection requires an opaque selector")
        selector.plan.selectors[selector.occurrence] = min(count, 2)
        indices = list(range(min(count, 2)))
        return indices[::-1] if reverse else indices
    raw = selector.detach().reshape(-1).tolist()
    indices = [int(value) for value in raw]
    if not indices or any(i != v or not 0 <= i < count for i, v in zip(indices, raw)) or len(set(indices)) != len(indices):
        raise ValueError("invalid expert selection")
    return indices[::-1] if reverse else indices

@dataclass(frozen=True)
class RepetitionInfo:
    instance_path: tuple
    index: int

def predicate_value(value):
    if isinstance(value, Tensor) and value.numel() == 1 and value.item() in (0, 1):
        return bool(value.item())
    raise TypeError('choice selector must return a one-element tensor containing 0 or 1')

class WeightBank(nn.Module):
    def __init__(self, declared_shape=None, data_size=''):
        super().__init__()
        self.values = nn.ParameterDict()
        self.declared_shape = declared_shape
        self.data_size = data_size

    @staticmethod
    def key(instance_path):
        return "root" if not instance_path else "_".join(str(i) for i in instance_path)

    def set(self, instance_path, value):
        self.values[self.key(instance_path)] = nn.Parameter(value)

    def update(self, instance_path, value):
        parameter = self.at(instance_path)
        if isinstance(value, _Shape):
            parameter.plan.match(parameter.rows, value.rows)
            parameter.plan.match(parameter.cols, value.cols)
            return parameter
        if parameter.shape != value.shape:
            raise ValueError('weight update shape mismatch')
        with torch.no_grad():
            parameter.copy_(value)
        return parameter

    def at(self, instance_path):
        if hasattr(self, '_shape_plan'):
            key = self.key(instance_path)
            if key not in self._demo_shapes:
                self._demo_shapes[key] = self._shape_plan.tensor()
                if self.declared_shape is not None:
                    _declared(self, self._demo_shapes[key], 'weight ' + key, self.declared_shape)
            return self._demo_shapes[key]
        return self.values[self.key(instance_path)]

class StateBank(nn.Module):
    """Per-instance tensor state, movable with the model but not checkpoint weights."""
    def __init__(self):
        super().__init__()
        self._shape_values = {}

    def at(self, state_path):
        key = WeightBank.key(state_path)
        if key in self._shape_values:
            return self._shape_values[key]
        if key not in self._buffers:
            raise RuntimeError(f'persistent tensor at {state_path} is uninitialized')
        return self._buffers[key]

    def set(self, state_path, value):
        key = WeightBank.key(state_path)
        if isinstance(value, _Shape):
            self._shape_values[key] = value
            return value
        self.register_buffer(key, value.clone(), persistent=False)
        return self._buffers[key]

    def clear(self):
        self._shape_values.clear()
        self._buffers.clear()
        self._non_persistent_buffers_set.clear()


def concat_state(values):
    if not values:
        raise ValueError('cannot append an empty state group')
    if isinstance(values[0], _Shape):
        first = values[0]
        rows = first.plan.dimension()
        equation = {rows: -1}
        for value in values:
            first.plan.match(first.cols, value.cols)
            equation[value.rows] = equation.get(value.rows, 0) + 1
        first.plan.equal(equation)
        return _Shape(first.plan, rows, first.cols)
    return torch.cat(values, dim=0)

class _ShapePlan:
    """Solve homogeneous dimension constraints for a small synthetic demo."""
    def __init__(self):
        self.dimensions = 0
        self.equations = {}
        self.selectors = {}
        self.opaque_shapes = {}
        self.opaque_calls = {}
        self.defaults = []
        self.has_declarations = False

    def dimension(self):
        result = self.dimensions
        self.dimensions += 1
        return result

    def tensor(self):
        return _Shape(self, self.dimension(), self.dimension())

    def match(self, left, right):
        if left != right:
            self.equal({left: 1, right: -1})

    def equal(self, equation):
        row = {key: Fraction(value) for key, value in equation.items() if value}
        while row:
            pivot = max(row)
            if pivot == -1:
                raise ValueError("incompatible declared tensor shapes")
            factor = row[pivot]
            if pivot not in self.equations:
                self.equations[pivot] = {key: value / factor for key, value in row.items()}
                return
            for key, value in self.equations[pivot].items():
                updated = row.get(key, 0) - factor * value
                if updated:
                    row[key] = updated
                else:
                    row.pop(key, None)

    def solve(self):
        # Preserve opaque input dimensions where explicit constraints allow it.
        for left, right in self.defaults:
            try:
                self.match(left, right)
            except ValueError:
                pass  # equal() only inserts a row after successful elimination.
        expressions = [{index: Fraction(1)} for index in range(self.dimensions)]
        constants = {-1: Fraction(1)}
        for pivot, row in sorted(self.equations.items()):
            expression = {}
            for key, value in row.items():
                if key != pivot:
                    for free, coefficient in (constants if key == -1 else expressions[key]).items():
                        expression[free] = expression.get(free, 0) - value * coefficient
            expressions[pivot] = {key: value for key, value in expression.items() if value}
        scales = {}
        for expression in expressions:
            for free, value in expression.items():
                if free == -1:
                    continue
                scales[free] = math.lcm(scales.get(free, 1), value.denominator)
        scales[-1] = 1
        sizes = [sum(value * scales[free] for free, value in expression.items()) for expression in expressions]
        if any(size <= 0 or size.denominator != 1 for size in sizes):
            raise ValueError("Cannot infer positive demo dimensions; supply explicit shapes and weights")
        return [int(size) for size in sizes]

class _Shape:
    def __init__(self, plan, rows, cols, occurrence=None):
        self.plan, self.rows, self.cols = plan, rows, cols
        self.occurrence = occurrence
        self.rank = 2
        self.singleton_rows = False
        self.singleton_cols = False

    def __matmul__(self, other):
        self.plan.match(self.rows if self.rank == 1 else self.cols, other.rows)
        result = _Shape(self.plan, other.cols if self.rank == 1 else self.rows,
                        self.cols if self.rank == 1 else other.cols)
        result.rank = self.rank
        result.singleton_cols = self.rank == 1
        return result

    def __add__(self, other):
        if self.rank != other.rank:
            def matrix(value):
                if value.rank == 2: return value
                result = _Shape(value.plan, value.cols, value.rows)
                result.singleton_rows = True
                result.singleton_cols = value.singleton_rows
                return result
            return matrix(self) + matrix(other)
        if not self.singleton_rows and not other.singleton_rows:
            self.plan.match(self.rows, other.rows)
        if not self.singleton_cols and not other.singleton_cols:
            self.plan.match(self.cols, other.cols)
        result = _Shape(self.plan, other.rows if self.singleton_rows else self.rows,
                        other.cols if self.singleton_cols else self.cols)
        result.rank = self.rank
        result.singleton_rows = self.singleton_rows and other.singleton_rows
        result.singleton_cols = self.singleton_cols and other.singleton_cols
        return result

    __sub__ = __add__

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise TypeError('shape multiplication requires a scalar')
        return self

    def transpose(self, first, second):
        result = _Shape(self.plan, self.cols, self.rows)
        result.singleton_rows, result.singleton_cols = self.singleton_cols, self.singleton_rows
        return result

    def size(self, sizes):
        return (sizes[self.rows],) if self.rank == 1 else (sizes[self.rows], sizes[self.cols])

def _declared(model, value, name, shape=None):
    if isinstance(value, list):
        return [_declared(model, member, name, shape) for member in value]
    shape = model._tensor_shapes[name] if shape is None else shape
    if isinstance(value, _Shape):
        if len(shape) not in (1, 2):
            raise ValueError(f'demo shape inference supports rank 1 or 2, got {name}: {shape}')
        value.plan.has_declarations = True
        value.rank = len(shape)
        value.singleton_rows = shape[0] == 1
        value.singleton_cols = len(shape) == 1 or shape[1] == 1
        value.plan.equal({value.rows: 1, -1: -shape[0]})
        value.plan.equal({value.cols: 1, -1: -(shape[1] if len(shape) == 2 else 1)})
    elif getattr(model, '_demo_shapes_ready', False) and tuple(value.shape) != shape:
        raise ValueError(f'{name}: expected declared shape {shape}, got {tuple(value.shape)}')
    return value

def _placeholder(model, value, instance_path, occurrence):
    key = (occurrence, tuple(instance_path))
    # A zero-input operation passes value=None; its declared destination fixes the shape.
    plan = value.plan if isinstance(value, _Shape) else getattr(model, '_shape_plan', None) if value is None else None
    # The same operation can see growing cache dimensions on successive calls.
    calls = plan.opaque_calls if plan is not None else getattr(model, '_demo_opaque_calls', None)
    if calls is not None:
        invocation = calls.get(key, 0)
        calls[key] = invocation + 1
        key = (*key, invocation)
    if plan is not None:
        if key not in plan.opaque_shapes:
            result = plan.tensor()
            result.occurrence = key
            plan.opaque_shapes[key] = result
            if value is not None:
                plan.defaults.extend([(result.rows, value.rows), (result.cols, value.cols)])
        return plan.opaque_shapes[key]
    if value is None:
        shapes = getattr(model, '_demo_opaque_shapes', {})
        if key not in shapes:
            raise NotImplementedError(f'{occurrence} has no inputs: implement it, or run the demo to get a placeholder')
        return torch.zeros(shapes[key])
    count = getattr(model, '_demo_selectors', {}).get(key)
    if count is not None:
        return torch.arange(count, device=value.device, dtype=value.dtype).reshape(1, -1)
    shape = getattr(model, '_demo_opaque_shapes', {}).get(key, tuple(value.shape))
    return torch.zeros(shape, device=value.device, dtype=value.dtype)

def run_demo(model, input_count, min_tokens=16):
    global _log_enabled
    if not isinstance(min_tokens, int) or min_tokens < 1:
        raise ValueError("min_tokens must be a positive integer")
    _log('Preparing demo shapes and random weights; operations are placeholders.')
    plan = _ShapePlan()
    if input_count != len(model._input_names):
        raise ValueError('input_count does not match the model entry')
    inputs, flat_inputs = [], []
    def input_tensor(name):
        value = plan.tensor()
        declaration = model._input_declarations.get(name)
        if declaration:
            _declared(model, value, declaration)
        flat_inputs.append(value)
        return value
    for name in model._input_names:
        record = model._input_types.get(name)
        if record is list:
            inputs.append([input_tensor(name) for _ in range(model._input_array_sizes[name])])
        elif record:
            inputs.append(record(**{field: input_tensor(name + '.' + field) for field in record.__dataclass_fields__}))
        else:
            inputs.append(input_tensor(name))
    banks = [module for module in model.modules() if isinstance(module, WeightBank)]
    states = [module for module in model.modules() if isinstance(module, StateBank)]
    for bank in banks:
        bank._shape_plan = plan
        bank._demo_shapes = {}
    model._shape_plan = plan  # zero-input operations draw their placeholders from it
    logging_enabled = _log_enabled
    try:
        # The shape-only pass follows the same scopes without printing duplicate progress.
        _log_enabled = False
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            model(*inputs)
        sizes = plan.solve()
        # Scale every dimension together to preserve all matrix/split constraints.
        scale = 1 if plan.has_declarations else max([1] + [(min_tokens + sizes[value.rows] - 1) // sizes[value.rows] for value in flat_inputs])
        sizes = [size * scale for size in sizes]
    finally:
        _log_enabled = logging_enabled
        for state in states:
            state._shape_values.clear()
        for bank in banks:
            del bank._shape_plan
        del model._shape_plan
    generator = torch.Generator().manual_seed(0)
    for bank in banks:
        for key, shape in bank._demo_shapes.items():
            rows, cols = shape.size(sizes)
            bank.values[key] = nn.Parameter(torch.randn(rows, cols, generator=generator) / math.sqrt(rows))
        del bank._demo_shapes
    model._demo_selectors = plan.selectors
    model._demo_opaque_shapes = {key: value.size(sizes) for key, value in plan.opaque_shapes.items()}
    model._demo_opaque_calls = {}
    model._demo_shapes_ready = True
    def materialize(value):
        if isinstance(value, list):
            return [materialize(member) for member in value]
        if isinstance(value, _Shape):
            return torch.randn(*value.size(sizes), generator=generator)
        return type(value)(**{field: materialize(item) for field, item in vars(value).items()})
    values = [materialize(value) for value in inputs]
    _log(f'Input shapes: {[value.size(sizes) for value in flat_inputs]}')
    model.eval()
    with torch.no_grad():
        result = model(*values)
    outputs = vars(result) if hasattr(result, '__dataclass_fields__') else dict(enumerate(result)) if isinstance(result, list) else {'output': result}
    for name, value in outputs.items():
        _log(f'Demo {name}: shape={tuple(value.shape)}, '
             f'finite={bool(torch.isfinite(value).all())}', always=True)
    return result

)PY";
}
