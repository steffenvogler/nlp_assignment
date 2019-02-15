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

path = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\sentences_with_sentiment.csv"

df = read_list(path)

print (df.head())
print (len(df))