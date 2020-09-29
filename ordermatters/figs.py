from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional, Any
import numpy as np

from ordermatters import configs


def make_example_fig(mat,
                     xlabel='nouns (slot 1)',
                     ylabel='next words (slot 2)'):
    fig, ax = plt.subplots(dpi=163)
    plt.title('', fontsize=5)

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              cmap=plt.get_cmap('cividis'),
              interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def make_info_theory_fig(data: List[List[List[float]]],
                         title: str,
                         x_axis_label: str,
                         y_axis_label: str,
                         x_ticks: List[int],
                         labels1: List[str],
                         labels2: List[str],
                         y_lims: Optional[List[float]] = None,
                         ):
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
    plt.title(title, fontsize=configs.Figs.title_font_size)
    ax.set_ylabel(y_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.set_xlabel(x_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['' if n + 1 != len(x_ticks) else f'{i:,}' for n, i in enumerate(x_ticks)],
                       fontsize=configs.Figs.tick_font_size)
    if y_lims:
        ax.set_ylim(y_lims)

    # plot
    lines = [[], []]  # 1 list for age-ordered and 1 for age-reversed
    for ys in data:
        for n, y in enumerate(ys):
            line, = ax.plot(x_ticks, y, '-', linewidth=2, color=f'C{n}')
            lines[n].append(line)

    # legend
    add_double_legend(lines, labels1, labels2)

    return fig, ax


def add_double_legend(lines_list, labels1, labels2, y_offset=-0.3, fs=configs.Figs.leg_font_size):

    # make legend 2
    lines2 = [l[0] for l in lines_list]
    leg2 = plt.legend(lines2,
                      labels2,
                      loc='upper center',
                      bbox_to_anchor=(0.5, y_offset + 0.1), ncol=2, frameon=False, fontsize=fs)

    # add legend 1
    # make legend 1 lines black but varying in style
    lines1 = [Line2D([0], [0], color='black', linestyle='-'),
              Line2D([0], [0], color='black', linestyle=':'),
              Line2D([0], [0], color='black', linestyle='--')][:len(labels1)]
    plt.legend(lines1,
               labels1,
               loc='upper center',
               bbox_to_anchor=(0.5, y_offset), ncol=3, frameon=False, fontsize=fs)

    # add legend 2
    plt.gca().add_artist(leg2)  # order of legend creation matters here


def plot_singular_values(ys: List[np.ndarray],
                         max_s: int,
                         font_size: int = 12,
                         figsize: Tuple[int] = (5, 5),
                         markers: bool = False,
                         label_all_x: bool = False):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=None)
    plt.title('SVD of simulated co-occurrence matrix', fontsize=configs.Figs.title_font_size)
    ax.set_ylabel('Singular value', fontsize=configs.Figs.ax_font_size)
    ax.set_xlabel('Singular Dimension', fontsize=configs.Figs.ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    x = np.arange(max_s) + 1  # num columns
    if label_all_x:
        ax.set_xticks(x)
        ax.set_xticklabels(x)
    # plot
    for n, y in enumerate(ys):
        ax.plot(x, y, label='toy corpus part {}'.format(n + 1), linewidth=2)
        if markers:
            ax.scatter(x, y)
    ax.legend(loc='upper right', frameon=False, fontsize=configs.Figs.ax_font_size)
    plt.tight_layout()
    plt.show()
