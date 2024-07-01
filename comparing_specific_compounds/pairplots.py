import os
from typing import List

import pandas as pd
from pkg_resources import resource_filename

from data_import import COMPS_OF_INTEREST, my_color_map, read_norm_data, IMPORTANT_COMPS, numbering

_inputs_path = resource_filename(__name__, 'inputs')

_temp_outputs_path = resource_filename(__name__, 'temp_outputs')

_output_path = resource_filename(__name__, 'outputs')


def kdes(offset: int):
    # Plot Areas along bottom, and densities on Y axis. Showing diffs with species.
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    df = df.rename(columns={'species': 'Species'})

    import matplotlib.pyplot as plt
    import seaborn as sns
    with sns.plotting_context("notebook", font_scale=3):
        fig, axes = plt.subplots(nrows=3, ncols=3)  # axes is 2d array (3x3)
        axes = axes.flatten()  # Convert axes to 1d array of length 9
        fig.set_size_inches(20, 20)
        for i, column in enumerate(COMPS_OF_INTEREST[9 * offset:9 * (offset + 1)], 1):
            plt.subplot(3, 3, i)
            if i == 3:
                kd = sns.kdeplot(df, x=column, hue="Species", fill=True, legend=True, palette=my_color_map)
            else:
                kd = sns.kdeplot(df, x=column, hue="Species", fill=True, legend=False, palette=my_color_map)
            kd.set(ylabel=None)

        fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical', size=50)
        plt.tight_layout()
        plt.savefig(os.path.join('outputs', f'comps_kdes_{offset}.jpg'), dpi=300)
        plt.close()


def boxes(comps_to_plot: List[str], tag: str, number: bool = False, legend: bool = True):
    print(tag)
    # For each compound, plot a box and whiskers separated by species.
    # The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined
    # to be “outliers” using a method that is a function of the inter-quartile range.
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    df = df.rename(columns={'species': 'Species'})

    comp_abbreviation_dict = {'N-Eicosanoylserotonin': 'N-Eicsero', 'Dimethoxycinnamoylcaffeoylquinic acid II': 'DiMeCinCafQui II',
                              '5-O Caffeoylquinic acid': '5-O CafQui',
                              '4,5-di-O-caffeoylquinic acid': '4,5-di-O CafQui',
                              '4-O-caffeoyl-3-O-ferroyloylquinic acid': '4-O-Caf-3-O-FerQui'}
    if not number:
        df = df.rename(columns=comp_abbreviation_dict)

    import matplotlib.pyplot as plt
    import seaborn as sns
    with sns.plotting_context("notebook", font_scale=3):
        nrows = 4
        ncols = 3
        figsize = (22, 25)
        if len(comps_to_plot) > 12:  # account for the extra 37th compound
            nrows = 5
            figsize = (22, 31.25)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)  # axes is 2d array (3x3)
        # axes = axes.flatten()  # Convert axes to 1d array of length 9

        if len(comps_to_plot) > 12:
            axes[-1, -1].axis('off')
            axes[-1, -2].axis('off')

        for i, column in enumerate(comps_to_plot, 1):

            plt.subplot(nrows, ncols, i)

            y_label = column
            if not number:
                if column in comp_abbreviation_dict:
                    y_label = comp_abbreviation_dict[column]
            kd = sns.boxplot(df, y=y_label, hue="Species", legend=True, gap=.2, linewidth=2, palette=my_color_map)

            # Get legend to plot
            legend_handles, legend_labels = kd.get_legend_handles_labels()
            plt.legend([], [], frameon=False)  # Hide individual legends
            kd.tick_params(bottom=False)
            if number:
                kd.set(ylabel=str(numbering[column]) + '    ')
                kd.set_ylabel(kd.get_ylabel(), rotation=0, fontweight='bold')
        # Create the legend
        if legend:
            if len(comps_to_plot) > 12:
                fig.legend(legend_handles, legend_labels, loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.088), prop={'size': 46})
            else:
                fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
        plt.tight_layout(rect=(0, 0.018, 1, 1))
        plt.savefig(os.path.join('outputs', f'comps_boxes_{tag}.jpg'), dpi=300)
        plt.close()


def ratio_comparison():
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    to_check = ['5-O-Ferruloylquinic acid']
    for c in to_check:
        canephora_median = canephora_data[c].median()
        arabica_median = arabica_data[c].median()
        sten_median = stenophylla_data[c].median()

        print(f' comparing {c}')
        print(f'Canephora median: {canephora_median}')
        print(f'Arabica median: {arabica_median}')
        print(f'Stenophylla median: {sten_median}')

        print(f'Stenophylla/Arabica ratio: {sten_median / arabica_median}')

    pass


if __name__ == '__main__':
    print(f'Number of compounds: {len(COMPS_OF_INTEREST)}')
    for _o in [0, 1]:
        boxes(COMPS_OF_INTEREST[12 * _o:12 * (_o + 1)], _o, number=True, legend=False)
    boxes(COMPS_OF_INTEREST[24:], 2, number=True, legend=True)  # Include that awkward one

    boxes(IMPORTANT_COMPS, 'important')
    boxes(IMPORTANT_COMPS, 'important_numbered', number=True)
    # ratio_comparison()
