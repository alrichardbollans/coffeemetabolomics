import os

import pandas as pd
from pkg_resources import resource_filename
from sklearn.preprocessing import StandardScaler

from data_import import read_norm_data, my_color_map, COMPS_OF_INTEREST, numbering

_output_path = resource_filename(__name__, 'outputs')


def do_hca_plot():
    # Note that for country clusters, see https://stackoverflow.com/questions/48173798/additional-row-colors-in-seaborn-cluster-map
    # Create a distance matrix using your chosen metric
    import seaborn as sns
    from matplotlib import pyplot as plt
    plt.tight_layout()

    s_plot = sns.clustermap(scaled_X.T, figsize=(10, 10), cmap='viridis',
                            method=cluster_method, metric=cluster_metric, col_colors=col_colors,
                            yticklabels=False, xticklabels=False, cbar_pos=(0.02, 0.8, 0.02, 0.18))

    s_plot.ax_col_dendrogram.legend(markers, my_color_map.keys(), numpoints=1, title='Species', bbox_to_anchor=(0.05, 0.98), fontsize='12',
                                    title_fontsize='12', markerscale=2)

    plt.savefig(os.path.join(_output_path, 'hca.jpg'), dpi=600)

    plt.close()
    plt.clf()
    plt.cla()


def do_hca_plot_with_ticks():
    # Note that for country clusters, see https://stackoverflow.com/questions/48173798/additional-row-colors-in-seaborn-cluster-map
    # Create a distance matrix using your chosen metric
    import seaborn as sns

    s_tick_plot = sns.clustermap(scaled_X.T, figsize=(10, 10), cmap='viridis',
                                 method=cluster_method, metric=cluster_metric, col_colors=col_colors,
                                 yticklabels=False, xticklabels=False, cbar_pos=(0.02, 0.8, 0.02, 0.18))

    reordered_labels = scaled_X.T.index[s_tick_plot.dendrogram_row.reordered_ind].tolist()
    use_ticks = [reordered_labels.index(label) + .5 for label in COMPS_OF_INTEREST]
    to_shift = [12, 10, 13, 36, 21, 29, 33, 34, 25, 26, 9, 14, 15, 16, 17]
    for comp in COMPS_OF_INTEREST:
        if numbering[comp] in to_shift:
            numbering[comp] = '    ' + str(numbering[comp])
    s_tick_plot.ax_heatmap.set(yticks=use_ticks, yticklabels=numbering.values())
    s_tick_plot.ax_heatmap.set_yticklabels(s_tick_plot.ax_heatmap.get_ymajorticklabels(), fontsize=8)

    s_tick_plot.ax_col_dendrogram.legend(markers, my_color_map.keys(), numpoints=1, title='Species', bbox_to_anchor=(0.05, 0.98), fontsize='12',
                                         title_fontsize='12', markerscale=2)
    plt.savefig(os.path.join(_output_path, 'hca_with_ticks.jpg'), dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    X = df.drop(columns='species')
    standard_scaler = StandardScaler().set_output(transform='pandas')
    scaled_X = standard_scaler.fit_transform(X)
    species = df['species']
    # species.name = 'Species'
    col_colors = species.map(my_color_map).tolist()  # Passing a list means no label is given on RHS.
    # # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in my_color_map.values()]
    # Plot HCA with:
    # Correlation metric: 1 - the Pearson product moment correlation i.e. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html
    # Complete linkage for calculating clusters: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

    cluster_metric = 'correlation'
    cluster_method = 'complete'
    do_hca_plot_with_ticks()
