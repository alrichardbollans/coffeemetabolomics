import os
from itertools import combinations

import pandas as pd
from matplotlib import pyplot as plt
from pkg_resources import resource_filename
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from statsmodels.graphics.gofplots import qqplot

from comparing_specific_compounds.pairplots import COMPS_OF_INTEREST
from data_import import read_norm_data, SPECIES

_output_path = resource_filename(__name__, 'outputs')



def hochberg_correction(df: pd.DataFrame, p_value_col: str):
    # Hochberg Method
    # Yosef Hochberg, ‘A Sharper Bonferroni Procedure for Multiple Tests of Significance’, Biometrika 75, no. 4 (1988): 800–802, https://doi.org/10.1093/biomet/75.4.800.


    # The adjustement is the same, but the usage is slightly different.
    new_df = df.sort_values(by=p_value_col)
    n = len(new_df.index)
    new_df.reset_index(inplace=True, drop=True)

    new_df['hochberg_adjusted_p_value'] = new_df.apply(lambda x: x[p_value_col] * (n - x.name), axis=1)

    return new_df

def tukey_instance(sp1: str, sp2: str, sp3: str, compound: str):
    # Tukey is calculated by CD
    pass


def plot_histogram(X, outpath: str, bins=10, title='Histogram', xlabel='Values', ylabel='Frequency'):
    """
    Plots a histogram for the given values X.

    """
    plt.figure(figsize=(8, 6))
    plt.hist(X, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(outpath)
    plt.close()


def qq_plot(X, outpath):
    """
    Creates a normal quantile-quantile (Q-Q) plot for the given sample X.

    """
    qqplot(X)

    # Set labels and title
    plt.title('Normal Quantile-Quantile (Q-Q) Plot')

    plt.savefig(outpath)

    plt.close()


def normality_check(vals):
    if len(vals) < 30:
        stat, p = shapiro(vals)
        if p < 0.05:
            return False
        else:
            return True
    else:
        return True


def cont_pvalue_instance(sp1: str, sp2: str, compound: str):
    ''' Calculate Welchs t test for two species and a compound'''

    sp1_values = data_df[data_df['species'] == sp1][compound].values
    sp2_values = data_df[data_df['species'] == sp2][compound].values

    if normality_check(sp1_values) and normality_check(sp2_values):
        stat, pvalue = ttest_ind(sp1_values, sp2_values, equal_var=False)
        test_used = 'Welch'
    else:
        stat, pvalue = mannwhitneyu(sp1_values, sp2_values)
        test_used = 'Mann Whitney'
    return stat, pvalue, test_used


def pvalues():
    out_cols = ['compound', 'species1', 'species2', 'stat', 'pvalue', 'test_used']
    out_df = pd.DataFrame(columns=out_cols)
    for comp in COMPS_OF_INTEREST:
        combins = combinations(SPECIES, 2)
        for comb in combins:
            sp1 = comb[0]
            sp2 = comb[1]
            stat, pvalue, test_used = cont_pvalue_instance(sp1, sp2, comp)
            instance_df = pd.DataFrame([[comp, sp1, sp2, stat, pvalue, test_used]],
                                       columns=out_cols)
            out_df = pd.concat([out_df, instance_df])

    return out_df


def do_plots():
    for sp in SPECIES:
        for comp in COMPS_OF_INTEREST:
            sp1_values = data_df[data_df['species'] == sp][comp].values
            if normality_check(sp1_values):
                out_dir = os.path.join(_output_path, 'normality_plots', 'passed')

            else:
                out_dir = os.path.join(_output_path, 'normality_plots', 'failed')

            plot_histogram(sp1_values, os.path.join(out_dir, f'{comp}_{sp}_histogram.jpg'))
            qq_plot(sp1_values, os.path.join(out_dir, f'{comp}_{sp}_qq.jpg'))


def main():
    pvalue_df = pvalues()
    pvalue_df.to_csv(os.path.join(_output_path, 'pvalues.csv'), index=False)
    pvalue_df = pd.read_csv(os.path.join(_output_path, 'pvalues.csv'))
    corrected_df = hochberg_correction(pvalue_df, 'pvalue')
    corrected_df.to_csv(os.path.join(_output_path, 'pvalues_corrected.csv'))
    do_plots()


if __name__ == '__main__':
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    data_df = pd.concat([arabica_data, canephora_data, stenophylla_data])

    main()
