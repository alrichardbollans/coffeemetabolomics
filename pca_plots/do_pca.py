import os.path
import random

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

from data_import import my_color_map, read_norm_data, numbering, COMPS_OF_INTEREST


def get_data_to_pca():
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    X = df.drop(columns='species')
    standard_scaler = StandardScaler().set_output(transform='pandas')
    scaled_X = standard_scaler.fit_transform(X)
    y = df['species'].values

    return scaled_X, y


def get_pca_data(n_comp=None):
    from sklearn.decomposition import PCA
    scaled_X, y = get_data_to_pca()

    pca = PCA(svd_solver='full', n_components=n_comp)
    pca.fit(scaled_X)
    print(f'PCA components: {pca.n_components_}. Explaining variance: {n_comp}')
    pca_df = pd.DataFrame(pca.transform(scaled_X), index=scaled_X.index, columns=['PC' + str(i + 1) for i in range(pca.n_components_)])
    return pca, pca_df, y


def plot_pca():
    # Plot the PCAs and variances
    pca, pca_df, y = get_pca_data()

    pca_df['species'] = y

    #### Bar plot of explained_variance
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_
    )

    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        c='red',
        label='Cumulative Explained Variance')

    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'pca_variance.jpg'), dpi=300)
    plt.close()
    #### Matrix plot to show general groupings
    import plotly.express as px
    labels = {
        'PC' + str(i): f"PC {i} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        pca_df.drop(columns=['species']),
        labels=labels,
        dimensions=['PC1', 'PC2', 'PC3', 'PC4'],
        color=pca_df["species"],
        color_discrete_map=my_color_map
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(margin=dict(l=30, r=10, t=10, b=10), legend=dict(title=None, font=dict(size=22), itemsizing='constant',
                                                                       yanchor="top",
                                                                       y=0.9,
                                                                       xanchor="left",
                                                                       x=0.7
                                                                       ), margin_r=10, margin_l=80, margin_b=70)

    fig.write_image("outputs/pcas_matrix.jpg", scale=5)  # for some reason points are removed with scale>6
    #### 3D plot to show general groupings
    seaborn.set_style("whitegrid", {'axes.grid': False})

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d', )

    for s in pca_df['species'].unique():
        ax.scatter(pca_df['PC1'][pca_df['species'] == s], pca_df['PC2'][pca_df['species'] == s], pca_df['PC3'][pca_df['species'] == s], label=s)

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', '3D_pca.jpg'), dpi=300)
    plt.close()
    import plotly.express as px
    fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3', color=pca_df['species'],
        title=f'Total Explained Variance: {0:.2f}%'
    )
    plt.close()


def plot_loadings(var1, var2, plot_nlargest: bool = True):
    import seaborn as sns
    from matplotlib import pyplot as plt

    pca_transformer, pca_df, y = get_pca_data()
    pca_df['Species'] = y

    sns.scatterplot(data=pca_df, x=var1, y=var2, hue="Species", palette=my_color_map)

    pca_df = pca_df.drop(columns=['Species'])
    useful_df = pd.DataFrame(pca_transformer.components_, columns=pca_transformer.feature_names_in_, index=pca_df.columns)
    useful_df = useful_df.loc[[var1, var2]]


    if plot_nlargest:
        # COmpute the variables that influence the given principal components the most.
        nlargest_lists = useful_df.apply(lambda x: x.abs().nlargest(5).index.tolist(), axis=1).tolist()
        nlargest = list(set(nlargest_lists[0] + nlargest_lists[1]))
        comps_to_plot = nlargest
        scale_factor =1000
    else:
        comps_to_plot = COMPS_OF_INTEREST
        scale_factor = 1000  # Given the number of comps, add scaling to make loadings visible
    shift_up = [23, 32, 15, 11]
    shift_left = [37]
    shift_right = [34, 35]
    shift_down = [7,26]
    for comp in comps_to_plot:
        x_val = useful_df[comp].loc[var1] * scale_factor
        y_val = useful_df[comp].loc[var2] * scale_factor
        plt.arrow(0, 0, x_val, y_val, color='grey', alpha=0.6, shape='full', head_width=0.02, head_length=0.015)
        xshift =0
        yshift =0

        if not plot_nlargest:
            name = numbering[comp]
            fontdict = {'size' : 8}

            if name in shift_up:
                yshift = 1.3
            if name in shift_down:
                yshift = -1.5
            if name in shift_left:
                xshift = -1.5
            if name in shift_right:
                xshift = 1.7
        else:
            name = comp
            fontdict = {'size' : 6}
            xshift = random.uniform(-2, 2)
            yshift = random.uniform(-2, 2)
        plt.text((x_val * 1.07)+xshift, (y_val * 1.05)+yshift, name, color='black', ha='center', va='center',font = fontdict )

    plt.tight_layout()
    dpi = 600
    if plot_nlargest:
        plt.savefig(os.path.join('outputs', f'nlargest_loadings_{var1}_{var2}.jpg'), dpi=dpi)
    else:
        plt.savefig(os.path.join('outputs', f'loadings_{var1}_{var2}.jpg'), dpi=dpi)

    plt.close()


def do_permanova():
    from skbio.stats.distance import permanova
    from skbio import DistanceMatrix

    # Ref Anderson, Marti J. “A new method for non-parametric multivariate analysis of variance.” Austral Ecology 26.1 (2001): 32-46.
    if not os.path.exists(os.path.join('outputs', 'permoanova')):
        os.mkdir(os.path.join('outputs', 'permoanova'))

    def prm_test(df_to_test, cols, tag):
        # Calculate Euclidean distance between samples based on columns
        # See https://scikit.bio/docs/dev/generated/skbio.stats.distance.permanova.html
        dissimilarity_matrix = DistanceMatrix.from_iterable(df_to_test[cols].values, metric=euclidean)
        # df = dissimilarity_matrix.to_data_frame()
        # Use the permanova function from scikit-bio
        result = permanova(dissimilarity_matrix, df_to_test['species'], permutations=1000)

        # Step 6: Interpret Results
        print(tag)
        print(result)  # Print PERMANOVA results
        result.to_csv(os.path.join('outputs', 'permoanova', f'{tag}.csv'))

    ## Run on pcas. Note can look at individual pcs, those that explain eg. 80% variances, or all pcs
    pca, pca_df, y = get_pca_data(0.8)
    pcs = pca_df.columns.tolist()
    pca_df['species'] = y

    # Do all groups
    prm_test(pca_df, pcs, 'all_groups_pcs')

    ## Run on components in plots
    pca, pca_df, y = get_pca_data(4)
    pcs = pca_df.columns.tolist()
    pca_df['species'] = y

    # Do all groups
    prm_test(pca_df, pcs, 'all_groups_4_pcs')

    # Do all pcs
    pca, pca_df, y = get_pca_data(None)
    pcs = pca_df.columns.tolist()
    pca_df['species'] = y

    # Do all groups
    prm_test(pca_df, pcs, 'all_groups_all_pcs')

    ### Check original data too
    ## Unscaled
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    compound_cols = df.columns.tolist()
    compound_cols.remove('species')

    prm_test(df, compound_cols, 'all_groups_original_compounds')

    ## scaled (probably not necessary)
    scaled_X, y = get_data_to_pca()
    scaled_X['species'] = y
    prm_test(scaled_X, compound_cols, 'all_groups_original_compounds_scaled')


if __name__ == '__main__':
    # plot_pca()
    # do_permanova()
    plot_loadings('PC1', 'PC2')
    plot_loadings('PC1', 'PC2', plot_nlargest=False)
    # plot_loadings(0, 1, plot_nlargest=False)
