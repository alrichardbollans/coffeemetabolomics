import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data_import import read_norm_data

TARGET_COLUMN = 'species'


def check_twinning(X):
    dup_df1 = X[X.duplicated(keep=False)]

    if len(dup_df1.index) > 0:
        dup_df1.to_csv('twins.csv')
        # raise ValueError(f'Twins present')
        print('WARNING: TWINS PRESENT: see twins.csv')


def basic_data_prep():
    ''' Read data and split into X,y
    '''
    given_data, transposed_data, arabica_data, canephora_data, stenophylla_data = read_norm_data()
    df = pd.concat([arabica_data, canephora_data, stenophylla_data])
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    check_twinning(X)
    return X, y


def do_basic_preprocessing(X: pd.DataFrame = None, train_index=None, test_index=None):
    ''' Get train, test sets and scale'''
    # use copies
    X_copy = X.copy(deep=True)
    # Note iloc select by position rather than index label
    X_train, X_test = X_copy.iloc[train_index], X_copy.iloc[test_index]

    if (train_index is not None and test_index is None) or (test_index is not None and train_index is None):
        raise ValueError

    standard_scaler = StandardScaler().set_output(transform='pandas')
    standard_scaler.fit(X_train)
    scaled_X_train = standard_scaler.transform(X_train)
    scaled_X_test = standard_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


class clf_scores:

    def __init__(self, name: str, clf_class, init_kwargs=None):
        self.name = name
        self.clf_class = clf_class
        self.accuracies = []
        self.y_reals = []
        self.fones = []
        self.init_kwargs = init_kwargs

    def get_clf_instance(self):
        if self.init_kwargs is not None:
            clf_instance = self.clf_class(**self.init_kwargs)
        else:
            clf_instance = self.clf_class()
        return clf_instance

    def add_cv_scores(self, X_train, y_train, X_test, y_test):

        clf_instance = self.get_clf_instance()

        clf_instance.fit(X_train, y_train)

        y_pred = clf_instance.predict(X_test)
        y_proba = clf_instance.predict_proba(X_test)[:, 1]

        nan_ind = len(np.where(np.isnan(y_proba))[0])
        if nan_ind > 0:
            raise ValueError(f'WARNING: {nan_ind} NaN predictions given by {self.name} model')

        self.y_reals.append(y_test)
        acc = accuracy_score(y_test, y_pred)
        if len(X_test.columns)>1 and acc ==0:
            print(f'y_test: {y_test}')
            print(f'y_pred: {y_pred}')
            print(X_test)
        self.accuracies.append(acc)


def output_boxplot(df_all: pd.DataFrame, df_ind: pd.DataFrame, out_file: str, y_title: str):
    def make_boxplot_df(df):
        scores_w_model = []
        for col in df.columns.tolist():
            if 'Unnamed' not in col:
                for val in df[col].values:
                    scores_w_model.append([val, int(col)])
        boxplot_df = pd.DataFrame(scores_w_model, columns=[y_title, 'Principal Components'])
        boxplot_df['Principal Components'] = boxplot_df['Principal Components'] + 1  # Rename to start with 1
        means = boxplot_df.groupby('Principal Components')[y_title].mean()
        return boxplot_df, means

    all_boxplot_df, all_means = make_boxplot_df(df_all)
    ind_boxplot_df, ind_means = make_boxplot_df(df_ind)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rc('font', size=13)
    fig, ax = plt.subplots()
    sns.boxplot(x='Principal Components', y=y_title, data=all_boxplot_df, showmeans=True)
    # X values seems to be position index rather than value so take away 1
    sns.lineplot(y=all_means.values, x=[c-1 for c in all_means.index.values], color='green', label='Cumulative')
    sns.lineplot(y=ind_means.values, x=[c-1 for c in ind_means.index.values], color='red', label='Individual PCs')
    # meanline.set(xlabel='Principal Components', ylabel=y_title)
    # meanline.set_ylim([0, 1])
    # plt.xticks(fontsize=5)#
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


def output_scores(models_with_all_PCAS, models_with_single_PCA, output_dir: str, filetag: str):
    acc_dict_all = {}

    for model in models_with_all_PCAS:
        acc_dict_all[model.name] = model.accuracies

    acc_df_all = pd.DataFrame(acc_dict_all)
    acc_df_all.describe().to_csv(os.path.join(output_dir, filetag + 'acc_means.csv'))
    acc_df_all.to_csv(os.path.join(output_dir, filetag + 'acc.csv'))

    acc_dict_ind = {}

    for model in models_with_single_PCA:
        acc_dict_ind[model.name] = model.accuracies

    acc_df_ind = pd.DataFrame(acc_dict_ind)
    acc_df_ind.describe().to_csv(os.path.join(output_dir, filetag + 'ind_acc_means.csv'))
    acc_df_ind.to_csv(os.path.join(output_dir, filetag + 'ind_acc.csv'))
    output_boxplot(acc_df_all, acc_df_ind, os.path.join(output_dir, filetag + 'accuracy_boxplot.jpg'), y_title='Accuracy')


if __name__ == '__main__':
    basic_data_prep()
