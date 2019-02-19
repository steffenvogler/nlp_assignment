######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# Helper tools for NLP analysis
#
######################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def read_list(path):
    df = pd.read_csv(path, encoding='latin-1', index_col=0, delimiter=';')
    return df

def df_transform_int(df):
    result_table = []
    i = 0
    for index, row in df.iterrows():
        pos_score = row['Positive']
        neg_score = row['Negative']
        neut_score = row['Neutral']
        i += 1
        if pos_score == 1:
            result_table.append(2)
            continue

        if neg_score == 1:
            result_table.append(0)
            continue

        if neut_score == 1:
            result_table.append(1)
            continue

        else:
            print ("corrupt annotation")
            print (i)
            break

    return result_table

def df_transform(df):
    result_table = []
    i = 0
    for index, row in df.iterrows():
        pos_score = row['Positive']
        neg_score = row['Negative']
        neut_score = row['Neutral']
        i += 1
        if pos_score == 1:
            result_table.append('Positive')
            continue

        if neg_score == 1:
            result_table.append('Negative')
            continue

        if neut_score == 1:
            result_table.append('Neutral')
            continue

        else:
            print ("corrupt annotation")
            print (i)
            break

    return result_table

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def downstream_analysis_lex(table_gt, table_pred, analysis_type, preprocess_string, class_names):
    #print (len(table_gt))
    #print (len(table_pred))
    cnf_matrix = confusion_matrix(table_gt, table_pred)
    accuracy = accuracy_score(table_gt, table_pred)
    if preprocess_string == "with_3_classes":
        recall = recall_score(table_gt, table_pred, average='micro')
    else:
        recall = recall_score(table_gt, table_pred)
    accuracy = str(np.round(accuracy, 4))
    recall = str(np.round(recall, 4))
    print("Accuracy: {m}; Recall: {n}".format(m=accuracy, n=recall))

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without norm (analysis type: {m} {n} preprocessing)'.format(
                              m=analysis_type, n=preprocess_string))
    plt.savefig(analysis_type + '_wo_norm_' + preprocess_string + "_preprocessing_accuracy_of_" + accuracy + ".png")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix (analysis type: {m} {n} preprocessing)'.format(
                              m=analysis_type, n=preprocess_string))
    plt.savefig(analysis_type + '_with_norm_' + preprocess_string + "_preprocessing_accuracy_of_" + accuracy + ".png")

    print("-------------------")

def downstream_analysis_ml(table_gt, table_result, analysis_type, preprocess_string, class_names):
    # df_pred = df_result[['Positive', 'Neutral', 'Negative']]
    table_pred = table_result
    cnf_matrix = confusion_matrix(table_gt, table_pred)
    accuracy = accuracy_score(table_gt, table_pred)
    if preprocess_string == "with_3_classes":
        recall = recall_score(table_gt, table_pred, average='micro')
    else:
        recall = recall_score(table_gt, table_pred)
    accuracy = str(np.round(accuracy, 4))
    recall = str(np.round(recall, 4))
    print("Accuracy: {m}; Recall: {n}".format(m=accuracy, n=recall))


    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without norm (analysis type: {m} {n} preprocessing)'.format(
                              m=analysis_type, n=preprocess_string))
    plt.savefig(analysis_type + '_wo_norm_' + preprocess_string + "_preprocessing_accuracy_of_" + accuracy + ".png")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix (analysis type: {m} {n} preprocessing)'.format(
                              m=analysis_type, n=preprocess_string))
    plt.savefig(
        analysis_type + '_with_norm_' + preprocess_string + "_preprocessing_accuracy_of_" + accuracy + ".png")

    print("-------------------")
