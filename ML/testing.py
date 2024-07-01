import os

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from ML import basic_data_prep, do_basic_preprocessing, clf_scores, output_scores

_output_path = 'outputs'
logit_init_kwargs = {'max_iter': 1000, 'solver': 'liblinear'}


def run_test_for_given_num_of_PCs(out_dir: str, num_components: int):
    _bias_model_dir = os.path.join(_output_path, out_dir)

    ### Data
    X, y = basic_data_prep()

    ### models
    models_with_all_pcas = []
    for k in range(len(X.index))[:num_components]:
        models_with_all_pcas.append(clf_scores(str(k), LogisticRegression,
                                               init_kwargs=logit_init_kwargs))
    models_with_individual_pcas = []
    for k in range(len(X.index))[:num_components]:
        models_with_individual_pcas.append(clf_scores(str(k), LogisticRegression,
                                                      init_kwargs=logit_init_kwargs))

    loo = LeaveOneOut()
    for i, (train_index, test_index) in enumerate(loo.split(X)):

        print(f'{i}th fold')
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaled_X_train, scaled_X_test = \
            do_basic_preprocessing(X=X,
                                   train_index=train_index,
                                   test_index=test_index)

        pca = PCA(svd_solver='full').set_output(transform='pandas')
        pca.fit(scaled_X_train)
        pca_X_train = pca.transform(scaled_X_train)
        pca_X_test = pca.transform(scaled_X_test)

        for model in models_with_all_pcas:
            cols = ['pca' + str(i) for i in range(int(model.name) + 1)]
            model.add_cv_scores(pca_X_train[cols], y_train, pca_X_test[cols], y_test)

        for model in models_with_individual_pcas:
            col = 'pca' + model.name
            model.add_cv_scores(pca_X_train[[col]], y_train, pca_X_test[[col]], y_test)

    output_scores(models_with_all_pcas, models_with_individual_pcas, _bias_model_dir, '')


if __name__ == '__main__':
    _X, _y = basic_data_prep()
    run_test_for_given_num_of_PCs('first_pcs', 10)
    run_test_for_given_num_of_PCs('', len(_X.index)-1)
