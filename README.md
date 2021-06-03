# Is this News Headline Funny? 
## Description
In Natural Language Processing, humour detection is a challenging task. The SemEval-2020 Task 7 [(Hossain et al. 2020)](https://arxiv.org/pdf/2008.00304.pdf) aims to detect humour in English news headlines from micro-edits. Early humour detection systems are based on traditional machine learning methods like support vector machines, decision trees and Naive Bayes classifiers [(Castro et al. 2016)](https://arxiv.org/pdf/1703.09527.pdf). Most recently, pre-trained language models based on Transformer [(Vaswani et al. 2018)](https://arxiv.org/pdf/1706.03762.pdf%EF%BC%89%E6%8F%8F%E8%BF%B0%E4%BA%86%E8%BF%99%E6%A0%B7%E5%81%9A%E7%9A%84%E5%8E%9F%E5%9B%A0%E3%80%82) have been used to detect humour in tweets and jokes [(Weller et al. 2019)](https://arxiv.org/pdf/1909.00252.pdf). In this project, we tackle the task using pretrained BERT [(Devlin et a. 2019)](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) combined with a Feed Forward Neural Network and with a Convolutional Neural Network. We compare its performance with a base line approach using GloVe embeddings and Bi-directional LSTMs. We then assess the importance of stop-word removal and train the model using XGBoost, Random Forest and Support Vector Regression (SVR). This project has been completed in collaboration with Sylvie Shi and Bastien Lecoeur as part of the Natural Language Processing module at Imperial. The paper of this project is available [here](https://github.com/Nasmasim/humour-detection/blob/main/paper/paper.pdf). 

## Dataset

The HUmicroedit dataset contains 9653 training, 2420 development and 3025 testing examples. The goal is to predict the funniness score of an edited headline in the ranges of 0 to 3, where 0 means not funny and 3 means very funny. Sample of micro-edited headlines, scored by 5 participants: 

| original headline | edited headline | meanGrade |
| --------------    | --------------  | --------  |
| US imposes metal tariffs on key *allies* | US imposes metal tariffs on key *holes* | 0.2 |
| Trump border wall : Texans receiving letters about their *land*  | Trump border wall : Texans receiving letters about their *barbecue* | 2.2 |
|The Nunes memo , explained with *diagrams*  | The Nunes memo , explained with *puppets* | 2.0 |

Downloading GloVe embeddings: 
``` 
!wget -nc http://nlp.stanford.edu/data/glove.6B.zip -O glove.6B.zip
!unzip -n glove.6B.zip 
```

## Project Structure 

| Notebook | Description | 
| -------- | ----------- |
| [run_analysis.ipynb](https://github.com/Nasmasim/humour-detection/blob/main/notebooks/run_analysis.ipynb) | Preprocessing steps and preliminary analysis on the humicroedit dataset, including stop words removal and N-gram frequency analysis. |
| [run_GloVe_biLSTM.ipynb](https://github.com/Nasmasim/humour-detection/blob/main/notebooks/run_GloVe_biLSTM.ipynb) | Baseline approach using GloVe word embeddings combined with a biLSTM to predict the funniness score of the edited news headlines |
|[run_BERT_CNN_FFNN.ipynb](https://github.com/Nasmasim/humour-detection/blob/main/notebooks/run_BERT_CNN_FFNN.ipynb)| pretrained BERT word embeddings combined with a CNN and FFNN outperforming the baseline approach. |
|[run_TFIDF_ML.ipynb](https://github.com/Nasmasim/humour-detection/blob/main/notebooks/run_TFIDF_ML.ipynb)| TF-IDF word verctorizer combined with Linear regression, XGBoost, Random Forest and SVR to predict the funniness score. |
