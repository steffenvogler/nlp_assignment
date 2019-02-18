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


| solution                | approach | accuracy (3-label) | accuracy (2-label) | processing time* [sec] | Link to source                                               |
| ----------------------- | -------- | ------------------ | ------------------ | ---------------------- | ------------------------------------------------------------ |
| VADER (NLTK)            | lexicon  | 51.8%              | 58.2%              | 1.5                    | [VADER](https://github.com/cjhutto/vaderSentiment)           |
| AFINN                   | lexicon  | 47.7%              | 56.1%              | 1.43                   | [AFINN](https://github.com/fnielsen/afinn)                   |
| NRC                     | lexicon  | 45.1%              | 51%                | 76.5                   | [NRC](http://sentiment.nrc.ca/lexicons-for-research/)        |
| Log Reg                 | ML       | 64.8%              | 90%                | 0.61                   | [LogReg](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| NGRAM                   | ML       | 66.6%              | 80-97.5%           | 0.57                   | [NGRAM](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| SVM                     | ML       | 64-85% (3 runs)    | 82.5-92.5%         | 0.59                   | [SVM](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/Sentiment%20Analysis%20Walkthrough%20Part%202.ipynb) |
| Multinomial Naive Bayes | ML       | 62.9%              | 75%                | 0.6                    | [MultiNB](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) |
| Complement Naive Bayes  | ML       | 72.2%              | 87-95%             | 0.6                    | [ComplNB](https://scikit-learn.org/stable/modules/naive_bayes.html) |
| Bernoulli Naive Bayes   | ML       | 62-72.2%           | 80-87.5%           | 0.5                    | [BernouNB](https://scikit-learn.org/stable/modules/naive_bayes.html) |

'*' .... for ML: processing time includes feature transformation, fitting, and prediction



# Exemplary visualization

![](\static\SVM_with_norm_with_bin_class_preprocessing_accuracy_of_0.875.png?raw=true)

Normalized confusion matrix with a SVM-classifier on positive/negative dataset. Acc = 87.5%.



# Checklist

 - (as of Feb 17, 2019)
 - [x] Start ReadMe
 - [x] Implement VADER
 - [x] Implement AFINN
 - [x] Implement NLTK
 - [x] Implement NRC
 - [x] Implement SVM
 - [x] Implement Naive Bayes
 - [x] Test other Bayes modalities
 - [x] do binary classification
 - [ ] (long-term): collect more annotated data; try ELMo, ULMFiT or <u>BERT</u>
 - [x] Do visualization
