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

# ARGUMENTS
PATH = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\sentences_with_sentiment.csv"
WITH_PREPROCESSING = True

# INPUT AND PARSING
df = read_list(PATH)
df_test = df['Sentence']
sum_positive = df['Positive'].sum()
sum_neutral = df['Neutral'].sum()
sum_negative = df['Negative'].sum()
print ("Positive sentences: {m}; Neutral sentences: {n}; Negative sentences: {o}; Total # of sentences: {p}"\
       .format(m = sum_positive, n = sum_neutral, o = sum_negative, p = len(df)))

# PRE-PROCESSING
if WITH_PREPROCESSING == True:
    

print (df_test.head())