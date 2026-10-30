"""Draw the SETEP paper figures as JPEGs next to the paper text."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
EVAL = sys.argv[2]

INK = '#1f1f1e'          # primary text
INK2 = '#5f5e5a'         # secondary text, axes
GRID = '#e4e2dc'
BLUE = '#2a78d6'         # categorical slot 1
ORANGE = '#eb6834'       # categorical slot 2
BLUE_FILL = '#e3eefb'
GRAY_FILL = '#f3f2ee'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.edgecolor': INK2})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, format='jpeg', pil_kwargs={'quality': 95},
                bbox_inches='tight', facecolor='white')
    print(path)


# Figure 1: toolchain overview -------------------------------------------
def box(ax, x, y, w, h, title, sub='', generated=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=BLUE_FILL if generated else GRAY_FILL,
                                ec=BLUE if generated else INK2, lw=1.0))
    ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), title, ha='center', va='center',
            fontsize=6.8, fontweight='bold', color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha='center', va='center', fontsize=5.8, color=INK2)


def arrow(ax, x0, y0, x1, y1, dashed=False):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, shrinkA=0, shrinkB=0,
                                mutation_scale=8, linestyle='--' if dashed else '-'))


fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.set_xlim(0, 17.6)
ax.set_ylim(-0.2, 6.4)
ax.set_aspect('equal')
ax.axis('off')
H = 1.0
W = 2.6
mid = 3.0
box(ax, 0.1, mid - H / 2, W, H, 'SETEP source', '.setep file', generated=False)
box(ax, 3.1, mid - H / 2, W, H, 'Front end', 'parse and check')
box(ax, 6.1, mid - H / 2, W, H, 'Shared IR', 'tensor program')
arrow(ax, 2.7, mid, 3.1, mid)
arrow(ax, 5.7, mid, 6.1, mid)
trunk = 9.0
ax.plot([8.7, trunk], [mid, mid], color=INK2, lw=0.9)
rows = [5.2, 3.0, 0.8]
ax.plot([trunk, trunk], [rows[-1], rows[0]], color=INK2, lw=0.9)
xs = 9.4
names = [('Python emitter', 'model + setep.py'),
         ('ONNX bridge', 'static graph'),
         ('Graph builder', 'ops, shapes, costs')]
for y, (t, sub) in zip(rows, names):
    arrow(ax, trunk, y, xs, y)
    box(ax, xs, y - H / 2, W, H, t, sub)
xt = 14.9
box(ax, xt, rows[0] - H / 2, W, H, 'PyTorch', 'execution', generated=False)
arrow(ax, xs + W, rows[0], xt, rows[0])
ax.text((xs + W + xt) / 2, rows[0] + 0.15, '+ handwritten impl', ha='center', va='bottom', fontsize=5.6, color=INK2)
box(ax, xt, rows[1] - H / 2, W, H, 'Runtime', 'onnxruntime', generated=False)
arrow(ax, xs + W, rows[1], xt, rows[1])
ax.text((xs + W + xt) / 2, rows[1] + 0.12, '+ weights file', ha='center', va='bottom', fontsize=5.6, color=INK2)
xp = 12.4
box(ax, xp, rows[2] - H / 2, 2.0, H, 'Planner', 'Chakra traces')
arrow(ax, xs + W, rows[2], xp, rows[2])
arrow(ax, xp + 2.0, rows[2], xt, rows[2])
box(ax, xt, rows[2] - H / 2, W, H, 'ASTRA-sim', 'distributed sim', generated=False)
ax.text(xp + 1.0, -0.1, 'processors, memory', ha='center', va='center', fontsize=6, color=INK2)
arrow(ax, xp + 1.0, 0.05, xp + 1.0, rows[2] - H / 2, dashed=True)

ax.add_patch(FancyBboxPatch((0.1, 5.75), 0.45, 0.32, boxstyle='round,pad=0,rounding_size=0.06', fc=BLUE_FILL, ec=BLUE, lw=0.8))
ax.text(0.7, 5.91, 'SETEP implementation', va='center', fontsize=6.5, color=INK2)
ax.add_patch(FancyBboxPatch((0.1, 5.2), 0.45, 0.32, boxstyle='round,pad=0,rounding_size=0.06', fc=GRAY_FILL, ec=INK2, lw=0.8))
ax.text(0.7, 5.36, 'input or external tool', va='center', fontsize=6.5, color=INK2)
save(fig, 'setep_fig1_toolchain.jpg')
plt.close(fig)

# Figure 2: per-step logit error vs Hugging Face ---------------------------
pt = json.load(open(os.path.join(EVAL, 'compare_hf.json')))['results']
ox = json.load(open(os.path.join(EVAL, 'onnx_gpt2.json')))['results']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for i, ax in enumerate(axes):
    steps = pt[i]['steps']
    xs = list(range(len(steps)))
    py = [s['max_abs'] for s in steps]
    on = [s['onnx_vs_hf_max_abs'] for s in ox[i]['steps']]
    assert len(on) == len(py)
    assert all(s['greedy_match'] for s in steps)
    assert all(s['onnx_token'] == s['hf_token'] for s in ox[i]['steps'])
    ax.plot(xs, py, color=BLUE, lw=2, marker='o', ms=4.5, mec='white', mew=0.8, label='PyTorch')
    ax.plot(xs, on, color=ORANGE, lw=2, marker='s', ms=4.5, mec='white', mew=0.8, label='ONNX')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e-3)
    ax.set_xticks(xs)
    ax.set_xticklabels(['P'] + [str(k) for k in xs[1:]], fontsize=7)
    ax.set_xlabel('step (P = prefill, then decode)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    n = pt[i]['n_prompt_tokens']
    pad = pt[i]['left_padding']
    ax.set_title(f'{n}-token prompt' + (f', {pad} left padding' if pad else ', no padding'),
                 fontsize=8.5, color=INK, loc='left')
    # Direct labels at the last step; the legend below repeats identity.
    if i == 1:
        ax.text(xs[-1] + 0.3, py[-1], 'PyTorch', color=INK, fontsize=7, va='center')
        ax.text(xs[-1] + 0.3, on[-1], 'ONNX', color=INK, fontsize=7, va='center')
axes[0].set_ylabel('max |logit difference|')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
save(fig, 'setep_fig2_gpt2_agreement.jpg')
