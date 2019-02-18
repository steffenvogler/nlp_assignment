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
from datetime import datetime



# ARGUMENTS
PATH = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\sentences_with_sentiment.csv"
WITH_PREPROCESSING = False
# Fill the appraoch-dictionary with all algorithms that you want to test
APPROACH_DICT = { 'ngram': ngram, 'SVM': svm, 'Multi_NaiveBayer': multi_nb} # {'vader': vader, 'nrc': nrc, 'afinn': afinn, 'LogReg': LogReg, 'ngram': ngram, 'SVM': svm, 'Multi_NaiveBayer': multi_nb, 'Compl_NaiveBayer': compl_nb, 'Bernou_NaiveBayer': bernou_nb}

# INPUT AND PARSING
df = read_list(PATH)
df_test = df['Sentence']
sum_positive = df['Positive'].sum()
sum_neutral = df['Neutral'].sum()
sum_negative = df['Negative'].sum()
print ("Positive sentences: {m}; Neutral sentences: {n}; Negative sentences: {o}; Total # of sentences: {p}"\
       .format(m = sum_positive, n = sum_neutral, o = sum_negative, p = len(df)))

# PRE-PROCESSING
preprocess_string = "with_3_classes"
if WITH_PREPROCESSING == True:
    preprocess_string = "with_bin_class"
    print (df.head())
    df = preprocess_df(df)
    df_gt = df[['Positive','Neutral', 'Negative']]
    table_gt = df_transform(df_gt)
    print ("length gt table:"+str(len(table_gt)))
    class_names = ['Positive', 'Neutral', 'Negative']

else:
    df_gt = df[['Positive','Neutral', 'Negative']]
    table_gt = df_transform(df_gt)
    class_names = ['Positive', 'Neutral', 'Negative']

for i in range(0, len(APPROACH_DICT)):
    start_time = datetime.now()
    print("-------------------")
    item = list(APPROACH_DICT.values())[i]
    analysis_type = str(list(APPROACH_DICT.keys())[i])
    print ("Type of analysis: {m} {n} preprocessing".format(m = analysis_type, n = preprocess_string))
    table_pred, table_gt_2, table_result  = item(df)
    if len(table_pred) > 0:
        downstream_analysis_lex(table_gt, table_pred, analysis_type, preprocess_string, class_names)
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
    else:
        if WITH_PREPROCESSING == True:
            class_names = ['Positive', 'Negative']
        downstream_analysis_ml(table_gt_2, table_result, analysis_type, preprocess_string, class_names)
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))