"""Parse and semantic errors: every source below must be rejected with the
quoted message, and nothing may be written for it."""
from common import arguments, setep, source

args = arguments()

REJECTED = [
    ('weight_bang', 'A=2%4@fp32 w=4%4@fp32 RUN(A){A >> *w! >> B}', 'remove ! from weight'),
    ('weight_bang_transposed', 'A=2%4@fp32 w=4%4@fp32 RUN(A){A >> *w!^ >> B}', 'remove ! from weight'),
    ('weight_bang_declared_array', 'A=2%4@fp32 w[2]=4%4@fp32 RUN(A){A >> REPEAT/2{*w!} >> B}', 'remove ! from weight'),
    ('scalar_group', 'n=4 A=1%4@fp32 F(X:Y){X+Y} RUN(A){A:n >> F()}', 'scalar cannot'),
    ('weight_call_group', 'A=1%4@fp32 w=1%4@fp32 RUN(A){A:w()}', 'weights cannot'),
    ('comma_computation', 'RUN(A){A, A+A}', 'expected'),
    ('comma_scope', 'RUN(A){A >> {X, X+X}}', 'expected'),
    ('conflicting_splits', 'A=1%4[2]@fp32 B=1%4@fp32 RUN(A){A >> C; C >> B}', 'conflicting declared'),
    ('conflicting_dimensions', 'A=1%4[2]@fp32 B=1%8[2]@fp32 RUN(A){A >> B}', 'conflicting declared'),
    ('shape_not_array', 'A=1%4[2]@fp32 RUN(A){A[]}', 'not an array'),
    ('shape_no_inferred_array', 'A=1%4[2]@fp32 RUN(A){A >> REPEAT/[]{X}}', 'requires an input array'),
    ('indivisible_shape', 'A=1%5[2]@fp32 RUN(A){A}', 'not divisible'),
    # A repetition must precede its braces; a trailing / is a syntax error.
    ('trailing_repetition', 'RUN(A){A >> {A}/ >> B}', 'expected "}" found "/"'),
    ('missing_array', 'RUN(A){A >> REPEAT/[]{A} >> B}', 'requires an input array'),
    ('counted_reduce', 'RUN(A){A >+[2]> B}', 'array reduction'),
    ('old_element', 'RUN(A){A#}', 'unhandled character'),
    ('bad_transfer', 'RUN(A){A >3> B}', 'expected'),
    ('group_reduce', 'RUN(A:B){A:B >+[]> C}', 'array reduction'),
    ('scatter_rank', 'RUN(A){A >> REPEAT/2[]{A} >+:> B}', 'array reduction'),
    ('missing_parentheses', 'F{} RUN(A){A >> F() >> B}', 'require parentheses'),
    ('zero_input_undeclared', 'A=1%4@fp32 RANDOM(){} RUN(A){RANDOM() >> R; A+R}', 'declared destination'),
    ('declared_input_count', 'PAIR(X:Y){} RUN(A){A >> PAIR() >> B}', 'takes 2 inputs'),
    ('selector_parameter', 'SEL(){} RUN(A){A >> CHOOSE?SEL(){A+A | A+A+A} >> B}', 'needs one parameter'),
    ('open_declaration', 'A[]=1%4@fp32 RUN(A){A}', 'explicit count'),
    ('group_definition', 'A:B=1%4@fp32 RUN(A){A}', 'names one value'),
    ('definition_flow', 'RUN(A){X=1%4@fp32 >> B}', 'names one value'),
    ('top_level_expression', 'A=1%4@fp32 A >> B RUN(A){A}', 'must be definitions'),
    ('untyped_array', 'n[3]=4 RUN(A){A}', 'typed tensor or weight'),
    ('declared_repetition', 'A=1%4@fp32 w[3]=4%4@fp32 RUN(A){A >> REPEAT/4{*w[]} >> B}', 'does not match repetition'),
    ('declared_annotation', 'A=1%4@fp32 w[3]=4%4@fp32 RUN(A){w[4]; A*w >> B}', 'conflicts with declaration'),
]
for name, text, message in REJECTED:
    path = source(args, name, text)
    for backend in ([], ['--python'], ['--chakra']):
        code, output = setep(args, path, *backend, check=False)
        assert code == 1 and message in output, (name, backend, output)
    assert not path.with_suffix('.py').exists() and not path.with_suffix('.chakra.json').exists(), name

# Repetition no longer accepts a choice count or a selector suffix.
for repetition in ('/|3', '\\|3', '/3?SEL()', '/3[]?SEL()'):
    path = source(args, 'removed_choice', 'SEL(A){} RUN(A){A >> REPEAT' + repetition + '{A+A} >> B}')
    assert setep(args, path, check=False)[0] != 0, repetition

# Accepted edge cases: declared parameters fix only the inputs; an unused
# computation is not called.
path = source(args, 'abstract_group', 'SPLIT(T){} RUN(A){A >> SPLIT() >> X:Y:Z; X+Y+Z}')
setep(args, path, '--python')
path = source(args, 'bare_call', 'F(){} RUN(A){A >> F >> B}')
setep(args, path, '--python')
assert 'self.F(' not in path.with_suffix('.py').read_text()
# ! still marks persistent tensors.
setep(args, source(args, 'state_bang', 'A=1%4@fp32 RUN(A){A >> K!; K!+A >> B}'), '--python')

# Command-line errors exit with 2.
assert setep(args, check=False)[0] == 2
assert setep(args, path, '--no-such-option', check=False)[0] == 2
assert setep(args, path, '--onnx', '--weights', 'w.npz', '--zero-weights', check=False)[0] == 2
code, output = setep(args, args.work / 'missing.setep', check=False)
assert code == 1 and 'cannot open' in output, output
print(f'{len(REJECTED)} rejected sources and the command-line errors behave as expected')
