######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# NLP_analysis_lib_collection
#
# VADER-CODE-FROM: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
######################################################
import pandas as pd

def vader_init():
    global analyser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()

def vader(df):
    vader_init()

    df_results = pd.DataFrame()
    result_table = []
    for index, row in df.iterrows():
        sentence = row['Sentence']
        score = analyser.polarity_scores(sentence)
        compound_value = score['compound']

        if compound_value > 0:
            result_dict = {'Positive': 1, 'Neutral': 0, 'Negative': 0}
            result_table.append('Positive')

        if compound_value < 0:
            result_dict = {'Positive': 0, 'Neutral': 0, 'Negative': 1}
            result_table.append('Negative')

        if compound_value == 0.0:
            result_dict = {'Positive': 0, 'Neutral': 1, 'Negative': 0}
            result_table.append('Neutral')

        df_results = df_results.append(result_dict, ignore_index=True)

    #print (df_results.head())
    #print (result_table)
    return df_results, result_table