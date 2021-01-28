import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, confusion_matrix, classification_report, multilabel_confusion_matrix, \
    top_k_accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hinge_loss
from sklearn.svm import LinearSVC
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import type_of_target

random_state = 1


def MLP(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', max_iter=100, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=.1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(X_train, y_train)

    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    y_pred = mlp.predict(X_test).tolist()
    y_true = y_test
    y_pred = list(map(int, y_pred))  # 转换格式
    y_true = list(map(int, y_true))

    print("precision:", metrics.precision_score(y_true, y_pred))
    print("recall_score:", metrics.recall_score(y_true, y_pred))
    print("f1:", metrics.f1_score(y_true, y_pred))
    sw = compute_sample_weight(class_weight='balanced', y=y_true)

    ans = classification_report(y_true, y_pred, sample_weight=sw, digits=3)
    print(ans)

    metrics.plot_roc_curve(mlp, X_test, y_test)
    plt.show()

    # rocprinting(mlp,X_train,X_test,y_train,y_test)
    # prprinting(mlp,X_train,X_test,y_train,y_test)

    """fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()"""


def prprinting(model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))


def rocprinting(model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import roc_auc_score, auc
    import matplotlib.pyplot as plt

    y_predict = model.predict(X_test)
    y_probs = model.predict_proba(X_test)  # 模型的预测得分

    y_test = list(map(int, y_test))
    y_predict = list(map(int, y_predict))  # 转换格式

    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), y_predict, pos_label=1)

    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    # 开始画ROC曲线
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()


def bijiao():
    print(__doc__)

    import warnings

    import matplotlib.pyplot as plt

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import datasets
    from sklearn.exceptions import ConvergenceWarning

    # different learning rate schedules and momentum parameters
    params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'adam', 'learning_rate_init': 0.01}]

    labels = ["constant learning-rate", "constant with momentum",
              "constant with Nesterov's momentum",
              "inv-scaling learning-rate", "inv-scaling with momentum",
              "inv-scaling with Nesterov's momentum", "adam"]

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'black', 'linestyle': '-'}]

    def plot_on_dataset(X, y, ax, name):
        # for each dataset, plot learning for each learning strategy
        print("\nlearning on dataset %s" % name)
        ax.set_title(name)

        X = MinMaxScaler().fit_transform(X)
        mlps = []
        if name == "digits":
            # digits is larger but converges fairly quickly
            max_iter = 15
        else:
            max_iter = 400

        for label, param in zip(labels, params):
            print("training: %s" % label)
            mlp = MLPClassifier(random_state=0,
                                max_iter=max_iter, **param)

            # some parameter combinations will not converge as can be seen on the
            # plots so they are ignored here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                        module="sklearn")
                mlp.fit(X, y)

            mlps.append(mlp)
            print("Training set score: %f" % mlp.score(X, y))
            print("Training set loss: %f" % mlp.loss_)
        for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # load / generate some toy datasets
    iris = datasets.load_iris()
    X_digits, y_digits = datasets.load_digits(return_X_y=True)
    data_sets = [(iris.data, iris.target),
                 (X_digits, y_digits),
                 datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
                 datasets.make_moons(noise=0.3, random_state=0)]

    for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
                                                        'circles', 'moons']):
        plot_on_dataset(*data, ax=ax, name=name)

    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.show()


def mmlp(X_train, X_test, y_train, y_test):
    from sklearn.datasets import make_classification
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import shuffle
    import numpy as np
    from sklearn.datasets import make_multilabel_classification
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    # X, y = make_multilabel_classification(n_classes=3, random_state=0)
    #clf = MultiOutputClassifier(
    #    SVC(C=10.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001,
    #        cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr',
    #        break_ties=False, random_state=None))

    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(sparse=False)
    #ans = enc.fit_transform(y_train)
    #decoded = enc.inverse_transform(ans)

    #print(ans)
    clf = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(20, 20,20), activation='tanh', max_iter=10, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=.1),n_jobs=-1)
    #clf=RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        clf.fit(X_train, enc.fit_transform(y_train))

    print(clf.score(np.array(X_test), enc.transform(y_test)))

    y_pred = []
    y_true = []

    for y in clf.predict(X_test):
        y = [bool(x) for x in y]
        y_pred.append(y)
    for y in y_test:
        y = [bool(x) for x in y]
        y_true.append(y)




    """
    y_pred = np.array(clf.predict(X_test))
    y_true = np.array(y_test)

    #sw = compute_sample_weight(class_weight='balanced', y=y_true)
    #k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    #scores = cross_validate(clf, X_train, y_train, cv=k_fold, scoring=['f1_weighted'])

    # ans = classification_report(y_true, y_pred, sample_weight=sw,digits=3)
    # print(ans)
    print(type_of_target(y_test))

    print(y_pred)
    print(y_pred.argmax(axis=1))

    cm = multilabel_confusion_matrix(y_true, y_pred)
    # cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), sample_weight=sw)
    print(cm)
    # metrics.f1_score(np.array(y_test), np.array(y_pred), average='micro',multi_class = 'ovo')"""

    #print("Training set score: %f" % clf.score(X_train, np.array(y_train)))
    # print("Test set score: %f" % clf.score(X_test, np.array(y_test)))
    """
    from sklearn.metrics import roc_curve

    #y_score = clf.predict(X_test)

    #print('accuracy:{}'.format(accuracy_score(y_test, y_score)))
    print('precision:{}'.format(precision_score(y_test, y_score, average='micro')))
    print('recall:{}'.format(recall_score(y_test, y_score, average='micro')))
    print('f1-score:{}'.format(f1_score(y_test, y_score, average='micro')))

    y_score_pro = clf.predict_proba(X_test)  # 形式二：各类概率值
    y_one_hot = label_binarize(y_train, np.arange(X_test))  # 转化为one-hot
    # AUC值
    auc = roc_auc_score(y_one_hot, y_score_pro, average='micro')

    # 画ROC曲线
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score_pro.ravel())  # ravel()表示平铺开来
    plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    auc = roc_auc_score(y_one_hot, y_score_pro, average='micro')
    """


def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    #svm=svm_cross_validation(X_train,y_train)
    svm = OneVsRestClassifier(SVC(C=10.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr',
              break_ties=False, random_state=None))
    svm.fit(X_train,y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        svm.fit(X_train, y_train)

    print("Training set score: %f" % svm.score(X_train, y_train))
    print("Test set score: %f" % svm.score(X_test, y_test))

    y_pred = svm.predict(X_test).tolist()
    y_true = y_test
    y_pred = list(map(int, y_pred))  # 转换格式
    y_true = list(map(int, y_true))
    #print("precision:", precision_score(y_true, y_pred, average="micro"))
    #print("recall_score:", recall_score(y_true, y_pred, average="micro"))
    #print("f1:", f1_score(y_true, y_pred,average="micro"))
    #sw = compute_sample_weight(class_weight='balanced', y=y_true)
    #ans = classification_report(y_true, y_pred, sample_weight=sw, digits=2)
    #ans = classification_report(y_true, y_pred,digits=3)
    #print(ans)

    #y_scores = svm.decision_function(X_test)
    #print("precision:", metrics.precision_score(y_true, y_pred))
    #print("recall_score:", metrics.recall_score(y_true, y_pred))
    #print("f1:", metrics.f1_score(y_true, y_pred))
    #print(top_k_accuracy_score(y_true, y_scores, k=3))

    #metrics.plot_roc_curve(svm, X_test, y_test)
    #plt.show()

    #sw = compute_sample_weight(class_weight='balanced', y=y_true)
    #cm = confusion_matrix(y_true, y_pred)
    #print(cm)

    #ans = classification_report(y_true, y_pred,digits=3)
    #print(ans)


def test(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    svc = SVC()
    parameters = [
        {
            'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf']
        }
    ]
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    best_model.predict(X_test)

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

