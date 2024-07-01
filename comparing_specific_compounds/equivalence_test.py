import os

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from scipy.stats import mannwhitneyu
from statsmodels.stats.weightstats import ttost_ind

from comparing_specific_compounds import normality_check, hochberg_correction
from data_import import read_norm_data, IMPORTANT_FLAVOUR_COMPS

_output_path = resource_filename(__name__, 'outputs')


def ttost_pvalue_instance(sp1: str, sp2: str, compound: str):
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    data_df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    sp1_values = data_df[data_df['species'] == sp1][compound].values
    sp2_values = data_df[data_df['species'] == sp2][compound].values
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_ind.html
    # Low and upp are WRT peak area values. Similarly with rtost.
    # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5502906/ for discussion of setting bounds

    # Set a smallest effect size of interest to be
    m1 = np.concatenate((sp1_values, sp2_values))
    val_mean = m1.mean()
    effect_size = val_mean * 0.25
    low = -(float(effect_size))
    upp = (float(effect_size))
    if normality_check(sp1_values) and normality_check(sp2_values):
        pvalue, lower_test, upper_test = ttost_ind(sp1_values, sp2_values, low=low, upp=upp, usevar='unequal')
        oneside_test_used = 'Welch'
    else:
        _, p_greater = mannwhitneyu(sp1_values + effect_size, sp2_values, alternative='greater')
        _, p_less = mannwhitneyu(sp1_values - effect_size, sp2_values, alternative='less')
        pvalue = max(p_greater, p_less)
        oneside_test_used = 'MannWhitney'

    # Upper and lower tests are equivalent to this:
    # __, _p_greater = ttest_ind(sp1_values + effect_size, sp2_values, alternative='greater', equal_var=False)
    # __, _p_less = ttest_ind(sp1_values - effect_size, sp2_values, alternative='less', equal_var=False)

    return pvalue, oneside_test_used


def pvalues():
    out_df = pd.DataFrame(columns=['compound', 'species1', 'species2', 'tost_pvalue', 'oneside_test'])
    sp1 = 'Arabica'
    sp2 = 'Stenophylla'
    for comp in IMPORTANT_FLAVOUR_COMPS:

        pvalue, ons_test = ttost_pvalue_instance(sp1, sp2, comp)
        instance_df = pd.DataFrame([[comp, sp1, sp2, pvalue, ons_test]],
                                   columns=['compound', 'species1', 'species2', 'tost_pvalue', 'oneside_test'])
        out_df = pd.concat([out_df, instance_df])

    return out_df


if __name__ == '__main__':
    results = pvalues()
    results.to_csv(os.path.join(_output_path, 'ttost_results.csv'), index=False)
    results = pd.read_csv(os.path.join(_output_path, 'ttost_results.csv'))
    corrected_df = hochberg_correction(results, 'tost_pvalue')
    corrected_df.to_csv(os.path.join(_output_path, 'ttost_results_corrected.csv'))
