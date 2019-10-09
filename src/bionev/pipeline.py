from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.svm import LinearSVC, SVC

from bionev.utils import *


def do_link_prediction(
    *,
    embeddings,
    original_graph,
    train_graph,
    test_pos_edges,
    save_model=None,
    classifier_type: Optional[str] = None,
):
    train_neg_edges = generate_neg_edges(original_graph, len(train_graph.edges()))
    # create a auxiliary graph to ensure that testing negative edges will not used in training
    g_aux = copy.deepcopy(original_graph)
    g_aux.add_edges_from(train_neg_edges)
    test_neg_edges = generate_neg_edges(g_aux, len(test_pos_edges))

    x_train, y_train = get_xy_sets(embeddings, train_graph.edges(), train_neg_edges)
    if classifier_type == 'SVM':
        clf = SVC(gamma='auto', probability=True)
    elif classifier_type == 'RF':
        clf = RandomForestClassifier(n_estimators=100, max_depth=2)
    elif classifier_type == 'EN':
        clf = SGDClassifier(loss="log", penalty="elasticnet")
    elif classifier_type is None or classifier_type == 'LR':
        clf = LogisticRegression(solver='lbfgs')
    else:
        raise ValueError(f'Invalid classifier_type: {classifier_type}')

    clf.fit(x_train, y_train)
    x_test, y_test = get_xy_sets(embeddings, test_pos_edges, test_neg_edges)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    y_pred = clf.predict(x_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    if save_model is not None:
        joblib.dump(clf, save_model)
    print('#' * 9 + ' Link Prediction Performance ' + '#' * 9)
    print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}')
    print('#' * 50)
    return auc_roc, auc_pr, accuracy, f1, mcc


def create_prediction_model(
    *,
    embeddings,
    original_graph,
    save_model=None,
    classifier_type=None,
):
    train_neg_edges = generate_neg_edges(original_graph, len(original_graph.edges()))
    x_train, y_train = get_xy_sets(embeddings, original_graph.edges(), train_neg_edges)
    if classifier_type == 'SVM':
        clf = SVC(gamma='auto', probability=True)
    elif classifier_type == 'RF':
        clf = RandomForestClassifier(n_estimators=100, max_depth=2)
    elif classifier_type == 'EN':
        clf = SGDClassifier(loss="log", penalty="elasticnet")
    else:
        clf = LogisticRegression(solver='lbfgs')
    clf.fit(x_train, y_train)
    if save_model is not None:
        joblib.dump(clf, save_model)


def do_node_classification(
    *,
    embeddings,
    node_list,
    labels,
    testing_ratio=0.2,
    save_model=None,
    classifier_type=None,
):
    x_train, y_train, x_test, y_test = split_train_test_classify(
        embeddings,
        node_list,
        labels,
        testing_ratio=testing_ratio,
    )

    # binarizer = MultiLabelBinarizer(sparse_output=True)
    # y_all = np.append(y_train, y_test)
    # binarizer.fit(y_all)
    # y_train = binarizer.transform(y_train).todense()
    # y_test = binarizer.transform(y_test).todense()
    if classifier_type == 'SVM':
        clf = LinearSVC()
    elif classifier_type == 'RF':
        clf = RandomForestClassifier(n_estimators=100, max_depth=2)
    elif classifier_type == 'EN':
        clf = SGDClassifier(loss="log", penalty="elasticnet")
    else:
        clf = LogisticRegression(solver='lbfgs')
    clf.fit(x_train, y_train)
    # y_pred_prob = model.predict_proba(x_test)

    # small trick : we assume that we know how many label to predict
    # y_pred = get_y_pred(y_test, y_pred_prob)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    if save_model is not None:
        joblib.dump(clf, save_model)

    print('#' * 9 + ' Node Classification Performance ' + '#' * 9)
    print(f'Accuracy: {accuracy:.3f}, Micro-F1: {micro_f1:.3f}, Macro-F1: {macro_f1:.3f}, MCC: {mcc:.3f}')
    print('#' * 50)
    return accuracy, micro_f1, macro_f1, mcc
