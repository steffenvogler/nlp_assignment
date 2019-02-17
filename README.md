# nlp_assignment

# README
<!-- TOC -->

- [README](#readme)
- [General](#general)
- [Results](#results)
- [Checklist](#Checklist)

<!-- /TOC -->

___

# General

Assignment of a NLP sentiment analysis. Along the project I will try to implement as many solutions as possible (given the time). At the end we can compare accuracies of each approach.

# Results


| solution                | approach | accuracy        | Link to source                                               |
| ----------------------- | -------- | --------------- | ------------------------------------------------------------ |
| VADER (NLTK)            | lexicon  | 51.8%           | [VADER](https://github.com/cjhutto/vaderSentiment)           |
| AFINN                   | lexicon  | 47.7%           | [AFINN](https://github.com/fnielsen/afinn)                   |
| NRC                     | lexicon  | 45.1%           | [NRC](http://sentiment.nrc.ca/lexicons-for-research/)        |
| Log Reg                 | ML       | 72.2%           | [LogReg](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| NGRAM                   | ML       | 77,7%           | [NGRAM](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| SVM                     | ML       | 64-85% (3 runs) | [SVM](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| Multinomial Naive Bayes | ML       | 64,8% (?)       | [MultiNB](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) |



# Checklist
 - (as of Feb 17, 2019)
 - [x] Start Readme
 - [x] Implement VADER
 - [x] Implement AFINN
 - [x] Implement NLTK
 - [x] Implement NRC
 - [x] Implement SVM
 - [x] Implement Naive Bayes
 - [x] Do visualization
