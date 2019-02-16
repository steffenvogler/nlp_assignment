######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# NLP_analysis_lib_collection
#
# Pre-processing According to: https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# VADER-CODE-FROM: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
# NRC-lexicon: http://sentiment.nrc.ca/lexicons-for-research/
# AFINN-CODE: https://github.com/fnielsen/afinn
# LogReg+NGRAM-Code: https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb
######################################################
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from util import df_transform_int

def svm(df):
    sentence_array = []
    for index, row in df[:].iterrows():
        print(row['Sentence'])
        sentence = row['Sentence']
        sentence_array.append(sentence)

    labels = df_transform_int(df)
    print(labels[:5])
    sentence_train, sentence_test, label_train, label_test = train_test_split(sentence_array, labels, test_size=0.20)
    print(len(sentence_train))

    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    SPACE = " "

    def preprocess_reviews(reviews):
        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

        return reviews

    sentence_train_clean = preprocess_reviews(sentence_train)
    sentence_test_clean = preprocess_reviews(sentence_test)

    stop_words = ['in', 'of', 'at', 'a', 'the']
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(sentence_train_clean)
    X = ngram_vectorizer.transform(sentence_train_clean)
    X_test = ngram_vectorizer.transform(sentence_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, label_train, train_size=0.75
    )

    for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_val, svm.predict(X_val))))

    # Accuracy for C=0.001: 0.88784
    # Accuracy for C=0.005: 0.89456
    # Accuracy for C=0.01: 0.89376
    # Accuracy for C=0.05: 0.89264
    # Accuracy for C=0.1: 0.8928

    final = LinearSVC(C=0.01)
    final.fit(X, label_train)
    print("Final Accuracy: %s"
          % accuracy_score(label_test, final.predict(X_test)))

    return [], label_test, final.predict(X_test)

def ngram(df):
    sentence_array = []
    for index, row in df[:].iterrows():
        print(row['Sentence'])
        sentence = row['Sentence']
        sentence_array.append(sentence)

    labels = df_transform_int(df)
    print(labels[:5])
    sentence_train, sentence_test, label_train, label_test = train_test_split(sentence_array, labels, test_size=0.20)
    print(len(sentence_train))

    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    SPACE = " "

    def preprocess_reviews(reviews):

        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

        return reviews

    sentence_train_clean = preprocess_reviews(sentence_train)
    sentence_test_clean = preprocess_reviews(sentence_test)

    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(sentence_train_clean)
    X = ngram_vectorizer.transform(sentence_train_clean)
    X_test = ngram_vectorizer.transform(sentence_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, label_train, train_size=0.75)

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_val, lr.predict(X_val))))

    final_ngram = LogisticRegression(C=0.5)
    final_ngram.fit(X, label_train)
    print("Final Accuracy: %s"
          % accuracy_score(label_test, final_ngram.predict(X_test)))

    return [], label_test, final_ngram.predict(X_test)



def LogReg(df):
    sentence_array = []
    for index, row in df[:].iterrows():
        print (row['Sentence'])
        sentence = row['Sentence']
        sentence_array.append(sentence)

    labels = df_transform_int(df)
    print (labels[:5])
    sentence_train, sentence_test, label_train, label_test = train_test_split(sentence_array, labels, test_size=0.20)
    print (len(sentence_train))


    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    SPACE = " "

    def preprocess_reviews(reviews):

        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

        return reviews

    sentence_train_clean = preprocess_reviews(sentence_train)
    sentence_test_clean = preprocess_reviews(sentence_test)

    baseline_vectorizer = CountVectorizer(binary=True)
    baseline_vectorizer.fit(sentence_train_clean)
    X_baseline = baseline_vectorizer.transform(sentence_train_clean)
    X_test_baseline = baseline_vectorizer.transform(sentence_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X_baseline, label_train, train_size=0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_val, lr.predict(X_val))))

    final_model = LogisticRegression(C=0.25)
    final_model.fit(X_baseline, label_train)
    print("Final Accuracy: %s"
          % accuracy_score(label_test, final_model.predict(X_test_baseline)))
    # Final Accuracy: 0.88128
    return [], label_test, final_model.predict(X_test_baseline)

def preprocess_df(df):
    #nltk.download('stopwords')
    from nltk.corpus import stopwords

    english_stop_words = stopwords.words('english')

    def remove_stop_words(corpus):
        removed_stop_words = []
        for review in corpus:
            removed_stop_words.append(
                ' '.join([word for word in review.split()
                          if word not in english_stop_words])
            )
        return removed_stop_words

    def get_stemmed_text(corpus):
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

    def get_lemmatized_text(corpus):
        #nltk.download('wordnet')
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

    for index, row in df[:1].iterrows():
        print (row['Sentence'])
        sentence = row['Sentence']
        sentence_clean = remove_stop_words(sentence)
        df.set_value(index, 'Sentence', sentence_clean)

    #print (df.head())
    return df



def afinn_init():
    from afinn import Afinn
    afinn = Afinn()
    return afinn



def afinn(df):
    afinn = afinn_init()

    df_results = pd.DataFrame()
    result_table = []
    for index, row in df.iterrows():
        sentence = row['Sentence']
        score = afinn.score(sentence)
        compound_value = score

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
    return  result_table, [], []




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
    return result_table, [], []



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
    return  result_table, [], []