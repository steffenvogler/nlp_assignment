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


# ARGUMENTS
PATH = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\sentences_with_sentiment.csv"
WITH_PREPROCESSING = False
APPROACH_DICT = {'SVM': svm} # {'vader': vader, 'nrc': nrc, 'afinn': afinn, 'LogReg': LogReg, 'ngram': ngram, 'SVM': svm}

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
    df = preprocess_df(df)
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
    table_pred, table_gt, table_result  = item(df)
    if len(table_pred) > 0:
        downstream_analysis_lex(table_gt, table_pred, analysis_type, preprocess_string, class_names)
    else:
        downstream_analysis_ml(table_gt, table_result, analysis_type, preprocess_string, class_names)