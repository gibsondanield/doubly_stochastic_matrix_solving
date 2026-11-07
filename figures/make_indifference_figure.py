"""Figure for the blog, section 3: why the 4-bottle probabilities could be solved by hand.

Example (same as test_solve_probability_matrix): 4 bottles, 2 detectors, codes 0..3.
We have measured one bit: bottle 0's detector-1 bit is 0, so bottle 0 has code 0 or 1.

Panel A - the probabilities, as worked out on paper.
Panel B - why that worked: indifference. Cells nothing can tell apart share a colour and
          must share a probability; row/column sums then leave one answer.
Panel C - what's really going on: count the consistent arrangements through each cell.

Run:  .venv/bin/python figures/make_indifference_figure.py
"""
import itertools
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

N = 4
# allowed[bottle, code]: bottle 0 must have bit 1 == 0 -> codes 0, 1 only
allowed = np.ones((N, N), dtype=bool)
allowed[0, [2, 3]] = False

# --- ground truth by enumerating every arrangement consistent with the measurement ---
arrangements = [p for p in itertools.permutations(range(N)) if all(allowed[b, c] for b, c in enumerate(p))]
counts = np.zeros((N, N), dtype=int)
for p in arrangements:
    for b, c in enumerate(p):
        counts[b, c] += 1
total = len(arrangements)
probs = [[Fraction(int(counts[b, c]), total) for c in range(N)] for b in range(N)]

# symmetry classes: cells with the same probability by indifference
cls = np.full((N, N), -1)
cls[0, [0, 1]] = 0             # bottle 0 between its two possible codes
cls[1:, [0, 1]] = 1            # other bottles, codes bottle 0 might be using
cls[1:, [2, 3]] = 2            # other bottles, codes bottle 0 can't be using
class_names = ['a', 'b', 'c']

# --- palette (categorical slots validated with dataviz validate_palette.js in both modes) ---
# usage: make_indifference_figure.py [light|dark]   (dark matches the black blog background)
THEME = sys.argv[1] if len(sys.argv) > 1 else 'light'
TALL = len(sys.argv) > 2 and sys.argv[2] == 'tall'   # stacked layout for phone screens
if THEME == 'dark':
    SURFACE, INK, INK2, MUTED = '#000000', '#f0efec', '#c3c2b7', '#898781'
    CAT = ['#3987e5', '#d95926', '#199e70']
    SEQ = ['#184f95', '#1c5cab', '#256abf', '#5598e7', '#9ec5f4']  # low -> high = recedes -> pops
    LIGHT_TEXT = set(SEQ[:4])                                     # white text on these
    IMPOSSIBLE = '#1f1f1e'
else:
    SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
    CAT = ['#2a78d6', '#eb6834', '#1baf7a']
    SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf']  # light -> dark
    LIGHT_TEXT = set(SEQ[3:])
    IMPOSSIBLE = '#f0efec'


def seq_color(v):
    return SEQ[min(int(v * (len(SEQ) - 1) / 0.5 + 0.5), len(SEQ) - 1)]  # 0.5 = strongest


def seq_text(face):
    return '#ffffff' if face in LIGHT_TEXT else '#0b0b0b'


code_labels = [f'{c}\n({format(c, "02b")})' for c in range(N)]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(*((3, 1) if TALL else (1, 3)), figsize=(5.6, 22) if TALL else (13.5, 6.6), facecolor=SURFACE)


def cell(ax, b, c, face, text, text_color, weight='normal'):
    ax.add_patch(FancyBboxPatch((c + 0.04, b + 0.04), 0.92, 0.92, boxstyle='round,pad=0,rounding_size=0.06',
                                facecolor=face, edgecolor=SURFACE, linewidth=2))
    ax.text(c + 0.5, b + 0.5, text, ha='center', va='center', color=text_color, fontsize=14, fontweight=weight)


def frame(ax, title, subtitle):
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect('equal')
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(N) + 0.5, code_labels, color=INK2, fontsize=10)
    ax.set_yticks(np.arange(N) + 0.5, [f'bottle {b}' for b in range(N)], color=INK2, fontsize=10)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_xlabel('code (year)', color=MUTED, fontsize=10, labelpad=6)
    ax.xaxis.set_label_position('top')
    ax.text(0, -1.45, title, color=INK, fontsize=13, fontweight='bold', va='bottom', transform=ax.transData)
    ax.text(0, -1.05, subtitle, color=INK2, fontsize=10, va='bottom', transform=ax.transData)


# A: the probabilities
ax = axes[0]
frame(ax, 'A. What I solved by hand', 'P(bottle has code), given bottle 0\'s bit 1 = 0')
for b in range(N):
    for c in range(N):
        p = probs[b][c]
        if p == 0:
            cell(ax, b, c, IMPOSSIBLE, '0', MUTED)
        else:
            face = seq_color(float(p))
            cell(ax, b, c, face, str(p), seq_text(face))

# B: indifference classes
ax = axes[1]
frame(ax, 'B. Why: indifference', 'cells nothing can tell apart must be equal')
for b in range(N):
    for c in range(N):
        k = cls[b, c]
        if k < 0:
            cell(ax, b, c, IMPOSSIBLE, '0', MUTED)
        else:
            cell(ax, b, c, CAT[k], class_names[k], '#ffffff', weight='bold')
eqs = ['row 0:     a + a = 1         →  a = 1/2',
       'column 0:  a + 3b = 1        →  b = 1/6',
       'row 1:     2b + 2c = 1       →  c = 1/3']
for i, e in enumerate(eqs):
    ax.text(0, N + 0.55 + i * 0.42, e, color=INK, fontsize=10, family='DejaVu Sans Mono', va='top')

# C: arrangement counts
ax = axes[2]
frame(ax, 'C. What\'s really going on', f'arrangements through each cell (of {total})')
for b in range(N):
    for c in range(N):
        n = counts[b, c]
        if n == 0:
            cell(ax, b, c, IMPOSSIBLE, '0', MUTED)
        else:
            face = seq_color(n / total)
            cell(ax, b, c, face, f'{n}', seq_text(face))
ax.text(0, N + 0.55, f'P = (arrangements via cell) / {total}', color=INK, fontsize=10,
        family='DejaVu Sans Mono', va='top')
ax.text(0, N + 0.97, '= perm(minor) / perm(allowed)', color=INK, fontsize=10,
        family='DejaVu Sans Mono', va='top')
ax.text(0, N + 1.39, 'symmetry pinned it here; usually it won\'t', color=INK2, fontsize=10, va='top')

if TALL:
    fig.subplots_adjust(left=0.2, right=0.97, top=0.89, bottom=0.085, hspace=0.9)
else:
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.22, wspace=0.42)
out = Path(__file__).with_name('indifference_4x4' + ('_dark' if THEME == 'dark' else '') + ('_tall' if TALL else ''))
fig.savefig(out.with_suffix('.png'), dpi=200, facecolor=SURFACE)
fig.savefig(out.with_suffix('.svg'), facecolor=SURFACE)

# sanity: the hand answer equals the counting answer
assert probs[0][0] == Fraction(1, 2) and probs[1][0] == Fraction(1, 6) and probs[1][2] == Fraction(1, 3)
print('wrote', out.with_suffix('.png'), 'and .svg;', total, 'consistent arrangements')
