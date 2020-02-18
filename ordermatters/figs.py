from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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


def add_double_legend(lines_list, labels1, labels2, y_offset=-0.3, fs=12):

    # make legend 2
    lines2 = [l[0] for l in lines_list]
    leg2 = plt.legend(lines2,
                      labels2,
                      loc='upper center',
                      bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False, fontsize=fs)

    # add legend 1
    # make legend 1 lines black but varying in style
    lines1 = [Line2D([0], [0], color='black', linestyle='-'),
              Line2D([0], [0], color='black', linestyle=':'),
              Line2D([0], [0], color='black', linestyle='--')][:len(labels1)]
    plt.legend(lines1,
               labels1,
               loc='upper center',
               bbox_to_anchor=(0.5, y_offset + 0.1), ncol=3, frameon=False, fontsize=fs)

    # add legend 2
    plt.gca().add_artist(leg2)  # order of legend creation matters here


