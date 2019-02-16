######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# NLP_analysis_main
#
######################################################

from util import *
from analysis import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ARGUMENTS
PATH = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\sentences_with_sentiment.csv"
WITH_PREPROCESSING = False
APPROACH_DICT = {'nrc': nrc} # {'vader': vader, 'nrc': nrc}

# INPUT AND PARSING
df = read_list(PATH)
df_test = df['Sentence']
sum_positive = df['Positive'].sum()
sum_neutral = df['Neutral'].sum()
sum_negative = df['Negative'].sum()
print ("Positive sentences: {m}; Neutral sentences: {n}; Negative sentences: {o}; Total # of sentences: {p}"\
       .format(m = sum_positive, n = sum_neutral, o = sum_negative, p = len(df)))

# PRE-PROCESSING
preprocess_string = "without"
if WITH_PREPROCESSING == True:
    preprocess_string = "with"
    pass

# ANALYSIS
df_gt = df[['Positive','Neutral', 'Negative']]
table_gt = df_transform(df_gt)
class_names = ["Positive", "Neutral", "Negative"]

for i in range(0, len(APPROACH_DICT)):
    print("-------------------")
    item = list(APPROACH_DICT.values())[i]
    analysis_type = str(list(APPROACH_DICT.keys())[i])
    print ("Type of analysis: {m} {n} preprocessing".format(m = analysis_type, n = preprocess_string))
    table_pred = item(df)
    print (len(table_pred))
    #df_pred = df_result[['Positive', 'Neutral', 'Negative']]

    cnf_matrix = confusion_matrix(table_gt, table_pred)
    accuracy = accuracy_score(table_gt, table_pred)
    accuracy = str(np.round(accuracy, 4))
    print ("Accuracy: "+ accuracy)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without norm (analysis type: {m} {n} preprocessing)'.format(m = analysis_type, n = preprocess_string))
    plt.savefig(analysis_type+'_wo_norm_'+preprocess_string+"_preprocessing_accuracy_of_"+accuracy+".png")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix (analysis type: {m} {n} preprocessing)'.format(m = analysis_type, n = preprocess_string))
    plt.savefig(analysis_type+'_with_norm_'+preprocess_string+"_preprocessing_accuracy_of_"+accuracy+".png")

    print ("-------------------")
