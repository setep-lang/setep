"""Numerical PyTorch checks of the generated Python for small SETEP programs,
plus a Chakra export of each one (partitioned programs that the operation
graph cannot lower must be rejected with a clear message)."""
import importlib.util
import sys

from common import arguments, require_modules, setep, source

require_modules('torch')
import torch

args = arguments()
sys.path.insert(0, str(args.work.resolve()))


def generate(name, text):
    path = source(args, name, text)
    setep(args, path, '--python')
    spec = importlib.util.spec_from_file_location(name, path.with_suffix('.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def run(*command):
    import subprocess
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check(name, text, expected, cpp_body=None, python_model=None, cpp_model='', rows=1, partitioned=False,
          explicit_arrays=False, partition_supported=False):
    """cpp_body/cpp_model are kept for the shape of the original fixtures only."""
    module, path = generate(name, text)
    model = python_model(module) if python_model else module.Model()
    tensor = torch.arange(1, rows * 4 + 1, dtype=torch.float32).reshape(rows, 4)
    result = model(tensor)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float32).reshape(rows, -1))
    chakra = ['--chakra', '--nodes', '1', '--processors-per-node', '4', '--memory-mb', '1000']
    if partitioned:
        generated = path.with_suffix('.py').read_text()
        if not explicit_arrays:
            assert 'for element_' not in generated
        if partition_supported:
            setep(args, path, *chakra)
            return
        code, output = setep(args, path, *chakra, check=False)
        assert code != 0 and 'shape partition lowering is not implemented' in output, output
        return
    setep(args, path, *chakra)
    assert path.with_suffix('.0.et').stat().st_size > 0


check('named_scope', 'A=1%4@fp32 RUN(A){A >> TOKEN; TOKEN >> TOKENS; REPEAT/2{TOKEN >> TOKEN+TOKEN >> TOKEN; TOKENS:TOKEN >:> TOKENS}; TOKENS >> B}', [1,2,3,4,2,4,6,8,4,8,12,16],
      'Model model; check(model.forward(x).data==std::vector<float>({1,2,3,4,2,4,6,8,4,8,12,16}),"named scope");')
check('named_scope_group', 'A=1%4@fp32 RUN(A){A >> TOKEN; TOKEN >> TOKENS; REPEAT/2{TOKEN >> TOKEN+TOKEN >> TOKEN; TOKENS:TOKEN >:> TOKENS; TOKEN:TOKENS} >> TOKEN:TOKENS; TOKENS >> B}', [1,2,3,4,2,4,6,8,4,8,12,16],
      'Model model; check(model.forward(x).data==std::vector<float>({1,2,3,4,2,4,6,8,4,8,12,16}),"named scope group");')

check('named_nested', 'A=1%4@fp32 RUN(A){A >> TOKEN; REPEAT/2{TOKEN >> TOKEN+TOKEN >> TOKEN; REPEAT/2{TOKEN >> TOKEN+TOKEN >> TOKEN}}; TOKEN >> B}', [64,128,192,256],
      'Model model; check(model.forward(x).data==std::vector<float>({64,128,192,256}),"nested named scope");')
check('named_once', 'A=1%4@fp32 RUN(A){A >> TOKEN; {TOKEN+TOKEN >> TOKEN}; TOKEN >> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"named scope without repetition");')
check('unequal_concat', 'A=1%4@fp32 RUN(A){A+A >> C; A:C >:> LONG; A:LONG >:> B}', [1,2,3,4,1,2,3,4,2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({1,2,3,4,1,2,3,4,2,4,6,8}),"unequal widths");')

check('reverse_peers', r'A=1%4@fp32 RUN(A){A >[2]> P; REPEAT\2[]{P[]+P[]} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"reverse peers");')

# Repeated opaque calls may have different shapes as a sequence grows.
_, growing_path = generate('growing_demo',
    'A=1%4@fp32 BIAS(X){} RUN(A){A >> X; REPEAT/3{X >> BIAS() >> B; X+B >> X; X:A >:> X}; X}')
assert 'finite=True' in run(sys.executable, growing_path.with_suffix('.py'))

check('arithmetic', 'A=1%4@fp32 RUN(A){A >> {A+A-A} >> B}', [1,2,3,4],
      'Model model; check(model.forward(x).data==x.data,"arithmetic");')
check('split_merge', 'A=1%4@fp32 RUN(A){A >[2]> P; REPEAT/2[]{P[]+P[]} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"split merge");')
check('reduce', 'A=1%4@fp32 RUN(A){A >> REPEAT/3[]{A+A} >+[]> B}', [6,12,18,24],
      'Model model; check(model.forward(x).data==std::vector<float>({6,12,18,24}),"reduce");')
check('array_math', 'A=1%4@fp32 RUN(A){A >[2]> P; P+P >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"array math");')
check('inferred_forward', 'A=1%4@fp32 RUN(A){A >[2]> REPEAT/[]{X; X+X} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"inferred forward");')
check('inferred_reverse', r'A=1%4@fp32 RUN(A){A >[2]> REPEAT\[]{X; X+X} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"inferred reverse");')
check('grouped_arrays', 'A=1%4@fp32 RUN(A){A >[2]> P; A >[2]> Q; P:Q >> REPEAT/[]{X:Y; X+Y} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"grouped arrays");')
check('group_concat_arrays', 'A=1%4@fp32 RUN(A){A >[2]> P; P:P >:> Q; Q+Q >[]> B}', [2,4,2,4,6,8,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,2,4,6,8,6,8}),"group concat arrays");')
check('element_group', 'A=1%4@fp32 ADD(X:Y){X+Y} RUN(A){A >[2]> P; REPEAT/2[]{P[]:P[] >> ADD()} >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"element group");')
check('shape_partition', 'A=1%4[2]@fp32 F(X){X+X} RUN(A){A >> F() >> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"logical shape");', partitioned=True)
check('shape_group_call', 'A=1%4[2]@fp32 ADD(X:Y){X+Y} RUN(A){A:A >> ADD() >> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"logical group");', partitioned=True)
check('two_axis_partition', 'A=4[2]%4[2]@fp32 F(X){X+X} RUN(A){A >> F() >> B}', list(range(2,33,2)),
      'Model model; auto y=model.forward(x); check(y.rows==4 && y.cols==4,"logical axes"); for(size_t i=0;i<16;++i) check(y.data[i]==2*(i+1),"values");', rows=4, partitioned=True)
check('row_partition', 'A=4[2]%4@fp32 RUN(A){A+A >> B}', list(range(2,33,2)),
      'Model model; auto y=model.forward(x); check(y.rows==4 && y.cols==4,"row axis"); for(size_t i=0;i<16;++i) check(y.data[i]==2*(i+1),"values");', rows=4, partitioned=True, partition_supported=True)
check('nested_inference', 'A=4%4@fp32 RUN(A){A >[2]> REPEAT/[]{X >> REPEAT/2[]{X+X}} >[]> B}', [2,4,2,4,6,8,6,8,10,12,10,12,14,16,14,16,18,20,18,20,22,24,22,24,26,28,26,28,30,32,30,32],
      'Model model; auto y=model.forward(x); check(y.rows==4 && y.cols==8,"nested explicit arrays");', rows=4)
check('partition_with_explicit_array', 'A=1%4[2]@fp32 RUN(A){A >[2]> P; P+P >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"explicit split retained");', partitioned=True, explicit_arrays=True)
check('partition_transfer', 'n=2 A=1%4[n]@fp32 B=1%4[2]@fp32 RUN(A){A >> C; C >> B}', [1,2,3,4],
      'Model model; check(model.forward(x).data==x.data,"logical transfer");', partitioned=True)
check('local_shapes', 'A=1%4@fp32 RUN(A){A >> {n=4; B=1%n@fp16; A+A >> B} >> C}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"locals");')
check('group_access', 'A=1%4@fp32 RUN(A){A >> B; A:B >> {X:Y; X+Y} >> D}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"group access");')

check('group_function', 'A=1%4@fp32 F(LEFT:RIGHT){LEFT+RIGHT} RUN(A){A >> B; A:B >> F() >> D}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"group function");')
check('mapped_call', 'A=1%4@fp32 F(X){X >> X+X} RUN(A){A >[2]> P; P >> F() >[]> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"mapped call");')

check('array_function', 'A=1%4@fp32 F(){X[2] >+[]> Y} RUN(A){A >[2]> P; P >> F() >> B}', [4,6],
      'Model model; check(model.forward(x).data==std::vector<float>({4,6}),"array function");')
check('choice_pipeline', 'A=1%4@fp32 SEL(A){} RUN(A){A >> CHOOSE?SEL(){A+A >> X | A+A+A >> Y} >> B}', [3,6,9,12],
      'Selected model; check(model.forward(x).data==std::vector<float>({3,6,9,12}),"choice pipeline");',
      lambda module: type('Selected', (module.Model,), {'SEL': lambda self, choice_index, value, repetition=None: torch.tensor([int(choice_index == 1)])})(),
      'struct Selected : Model { Tensor SEL(int i,const Tensor&,const std::optional<RepetitionInfo>&) override { return Tensor(1,1,i==1); } };')

check('state_decode_loop', 'A=1%4@fp32 SAVE(A){A >> CACHE/2{A >> K!; A >> B}} STEP(A){A >> CACHE/2{X; K!+X >> K!; K! >> B}} RUN(A){A >> SAVE() >> REPEAT/3{STEP()} >> B}', [21,42,63,84],
      'Model model; check(model.forward(x).data==std::vector<float>({21,42,63,84}),"decode cache reuse");')

check('state', 'A=1%4@fp32 SAVE(A){A >> CACHE{A >> K!; A >> B}} STEP(A){A >> CACHE{K!+A >> K!; K! >> B}} RUN(A){A >> SAVE() >> STEP() >> B}', [2,4,6,8],
      'Model model; check(model.forward(x).data==std::vector<float>({2,4,6,8}),"persistent state");')
# RANDOM() takes no input; its declared destination R gives the result's shape.
check('zero_input', 'A=1%4@fp32 R=1%4@fp32 RANDOM(){} RUN(A){RANDOM() >> R; A+R}', [2,3,4,5],
      'Ones model; check(model.forward(x).data==std::vector<float>({2,3,4,5}),"zero-input source");',
      lambda module: type('Ones', (module.Model,), {'RANDOM': lambda self, **kwargs: torch.ones(1, 4)})(),
      'struct Ones : Model { Tensor RANDOM(const Path&,const std::string&) override { return Tensor(1,4,1); } };')
check('opaque_entry', 'A=1%4@fp32 RUN(A){}', [1,2,3,4],
      'Opaque model; check(model.forward(x).data==x.data,"opaque entry");',
      lambda module: type('Opaque', (module.Model,), {'RUN': lambda self, value, **kwargs: value})(),
      'struct Opaque : Model { Tensor RUN(const Tensor& value,const Path&,const std::string&) override { return value; } };')

check('array_results', 'A=1%4@fp32 F(X){X:X >[2]> Q:K} RUN(A){A >> F() >> P:R; P >+[]> B}', [4,6],
      'Model model; check(model.forward(x).data==std::vector<float>({4,6}),"array results");')

def choose(module):
    class Model(module.Model):
        def SEL(self, choice_index, value, repetition=None):
            return torch.tensor([int(choice_index == 1)])
    return Model()

selector = '''struct Selected : Model {
  Tensor SEL(int i,const Tensor&,const std::optional<RepetitionInfo>&) override { return Tensor(1,1,i==1?1:0); }
};'''
check('choice', 'A=1%4@fp32 SEL(A){} RUN(A){A >> CHOOSE?SEL(){A+A | A+A+A} >> B}', [3,6,9,12],
      'Selected model; check(model.forward(x).data==std::vector<float>({3,6,9,12}),"choice");', choose, selector)



def repeated_weights(module):
    model = module.Model()
    for i in range(12):
        model.weight_w.set((i,), torch.eye(4) * (2 if i == 7 else 1))
    return model

check('distinct_weights', 'A=1%4@fp32 w=4%4@fp32 RUN(A){w[12]; A >> BLOCK/12{*w[]} >> B}',
      [2,4,6,8],
      'Model model; for(int i=0;i<12;++i) { Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=i==7?2:1; model.weight_w.values[{i}]=w; } check(model.forward(x).data==std::vector<float>({2,4,6,8}),"distinct weights");', repeated_weights)

# w[12] = shape declares the weight array the w[12] annotation used to state.
check('declared_weights', 'A=1%4@fp32 w[12]=4%4@fp32 RUN(A){A >> BLOCK/12{*w[]} >> B}',
      [2,4,6,8],
      'Model model; for(int i=0;i<12;++i) { Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=i==7?2:1; model.weight_w.values[{i}]=w; } check(model.forward(x).data==std::vector<float>({2,4,6,8}),"declared weights");', repeated_weights)

def shared_weight(module):
    model = module.Model()
    model.weight_w.set((), torch.eye(4))
    return model

check('weight_partition', 'A=1%4[2]@fp32 w=4[2]%4@fp32 RUN(A){A*w >> B}', [1,2,3,4],
      'Model model; Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=1; model.weight_w.values[{}]=w; check(model.forward(x).data==x.data,"logical multiplication");', shared_weight, partitioned=True, partition_supported=True)

check('shared_weight', 'A=1%4@fp32 w=4%4@fp32 RUN(A){A >> REPEAT/12{*w} >> B}', [1,2,3,4],
      'Model model; Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=1; model.weight_w.values[{}]=w; check(model.forward(x).data==x.data,"shared weight");', shared_weight)

def group_weight(module):
    model = module.Model()
    model.weight_w.set((), torch.full((1, 4), 10.0))
    return model

group_setup = 'Model model; model.weight_w.values[{}]=Tensor(1,4,10); '
for name, expression, expected in (
    ('tensor_weight_group', 'A:w >> SUB()', [-9,-8,-7,-6]),
    ('weight_tensor_group', 'w:A >> SUB()', [9,8,7,6]),
    ('named_weight_group', 'A:w >> PAIR; PAIR >> SUB()', [-9,-8,-7,-6]),
):
    check(name, 'A=1%4@fp32 w=1%4@fp32 SUB(X:Y){X-Y} RUN(A){'+expression+' >> B}',
          expected, group_setup+'check(model.forward(x).data==std::vector<float>({' +
          ','.join(map(str,expected)) + '}),"mixed group");', group_weight)

# Group passing must preserve the Parameter object and its autograd connection.
for fixture in ('tensor_weight_group', 'weight_tensor_group', 'named_weight_group'):
    model = group_weight(sys.modules[fixture])
    parameter = model.weight_w.at(())
    model(torch.arange(1., 5.).reshape(1, 4)).sum().backward()
    assert model.weight_w.at(()) is parameter
    torch.testing.assert_close(parameter.detach(), torch.full((1, 4), 10.0))
    torch.testing.assert_close(parameter.grad, torch.full((1, 4), 1.0 if fixture == 'weight_tensor_group' else -1.0))

# H[3] = shape checks the count and element shape of the array bound to H.
check('declared_tensors', 'A=1%4@fp32 w=4%4@fp32 H[3]=1%4@fp32 RUN(A){A >> REPEAT/3[]{*w} >> H; H >+[]> B}',
      [3,6,9,12], 'Model model; Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=1; model.weight_w.values[{}]=w; check(model.forward(x).data==std::vector<float>({3,6,9,12}),"declared tensors");', shared_weight)

check('weight_group_header', 'A=1%4@fp32 w=1%4@fp32 SUB(X:Y){X-Y} RUN(A){A:w >> SUB() >> B}',
      [-9,-8,-7,-6], group_setup+'check(model.forward(x).data==std::vector<float>({-9,-8,-7,-6}),"header capture");', group_weight)
check('weight_group_update', 'A=1%4@fp32 w=1%4@fp32 SUB(X:Y){X-Y} RUN(A){A:A >> B:w; B:w >> SUB() >> C}',
      [0,0,0,0], group_setup+'check(model.forward(x).data==std::vector<float>({0,0,0,0}),"group weight update");', group_weight)
check('weight_group_transpose', 'A=1%4@fp32 w=4%4@fp32 MM(X:Y){X*Y} RUN(A){A:w^ >> MM() >> B}',
      [1,2,3,4], 'Model model; Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=1; model.weight_w.values[{}]=w; check(model.forward(x).data==x.data,"group transpose");', shared_weight)
check('indexed_weight_group', 'A=1%4@fp32 w=4%4@fp32 MM(X:Y){X*Y} RUN(A){w[12]; A >> REPEAT/12{X; X:w[] >> MM()} >> B}',
      [2,4,6,8], 'Model model; for(int i=0;i<12;++i) { Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=i==7?2:1; model.weight_w.values[{i}]=w; } check(model.forward(x).data==std::vector<float>({2,4,6,8}),"indexed group");', repeated_weights)

def reverse_weights(module):
    model = module.Model()
    for i in range(2):
        model.weight_w.set((i,), torch.eye(4) * (i + 2))
        model.weight_b.set((i,), torch.full((1, 4), float(2 - i)))
    return model

check('reverse', r'A=1%4@fp32 w=4%4@fp32 b=1%4@fp32 RUN(A){w[2]; b[2]; A >> REPEAT\2{*w[] +b[]} >> B}', [10,16,22,28],
      'Model model; for(int i=0;i<2;++i) { Tensor w(4,4); for(int j=0;j<4;++j) w.data[j*4+j]=i+2; model.weight_w.values[{i}]=w; model.weight_b.values[{i}]=Tensor(1,4,2-i); } check(model.forward(x).data==std::vector<float>({10,16,22,28}),"reverse");', reverse_weights)

for fixture in ('choice',):
    module = sys.modules[fixture]
    class Empty(module.Model):
        def SEL(self, choice_index, value, repetition=None):
            return torch.tensor([0])
    try:
        Empty()(torch.ones(1, 4))
        raise AssertionError('empty selection accepted')
    except ValueError as error:
        assert 'select' in str(error)

module = sys.modules['choice']
class Multiple(module.Model):
    def SEL(self, choice_index, value, repetition=None):
        return torch.tensor([1])
try:
    Multiple()(torch.ones(1, 4))
    raise AssertionError('multiple alternatives accepted')
except ValueError as error:
    assert 'exactly one' in str(error)

print('PyTorch numerical checks and Chakra exports passed')
