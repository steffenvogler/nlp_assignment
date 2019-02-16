######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# NLP_analysis_lib_collection
#
# VADER-CODE-FROM: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
# NRC-lexicon: http://sentiment.nrc.ca/lexicons-for-research/
######################################################
import pandas as pd
import nltk

def nrc(df):
    PATH_NRC_LEXICON = r"C:\Temp\DataScience_Team\DS_NLP_Assignment\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    df_lexicon = pd.read_csv(PATH_NRC_LEXICON, delimiter="\t", header=None)
    #print (df_lexicon.head())
    #nltk.download('punkt')
    result_table = []
    j=0
    for index, row in df[:].iterrows():
        sentence = row['Sentence']
        tokenized_text = nltk.word_tokenize(sentence)
        #print (tokenized_text)
        score = 0
        #print (j)
        j += 1
        for i in tokenized_text:
            try:
                #df_bool = df_lexicon[0].str.match(i, na=False)
                df_new = df_lexicon[df_lexicon[0] == i]
                hits = df_new[df_new[2] == 1]
                if hits[1].str.contains("negative").any() or hits[1].str.contains("sadness").any():
                    score -= 1
                elif hits[1].str.contains("positive").any() or hits[1].str.contains("joy").any():
                     score += 1
                else:
                    score = score
            except:
                score = score

        #print ("Sentiment score for this senntece: "+str(score))
        if score > 0:
            result_table.append("Positive")
        elif score < 0:
            result_table.append("Negative")
        else:
            result_table.append("Neutral")
    #print (result_table)
    return result_table

def vader_init():

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    return analyser

def vader(df):
    analyser = vader_init()

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
    return  result_table