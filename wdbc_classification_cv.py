import numpy as np
from sklearn import (datasets, tree, model_selection, naive_bayes, ensemble, svm)

if __name__ == '__main__':
    # Load a dataset
    wdbc = datasets.load_breast_cancer()

    # Train a model
    # TODO
    #1.GaussianNB
    #model = naive_bayes.GaussianNB(
    #    var_smoothing = 1e-06
    #)

    #2. DecisionTree
    #model = tree.DecisionTreeClassifier(
    #    max_depth = 7,
    #    min_samples_split = 10,
    #    min_samples_leaf = 20,
    #    criterion = 'entropy'
    #)

    #3. RandomForest
    model = ensemble.RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=7,
        random_state=42
    )
    #model = svm.SVC() - 11
    cv_results = model_selection.cross_validate(model, wdbc.data, wdbc.target, cv=5, return_train_score=True)

    # Evaluate the model
    acc_train = np.mean(cv_results['train_score'])
    acc_test = np.mean(cv_results['test_score'])
    print(f'* Accuracy @ training data: {acc_train:.3f}')
    print(f'* Accuracy @ test data: {acc_test:.3f}')
    print(f'* Your score: {max(10 + 100 * (acc_test - 0.9), 0):.0f}')