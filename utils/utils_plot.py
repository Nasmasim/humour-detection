import matplotlib.pyplot as plt
import seaborn as sns


def plot_sentece_length_stopwords(X, series):
    plt.plot(X, series[0],label="no stop words")
    plt.plot(X, series[1],label="simple")
    plt.plot(X, series[2],label="custom")
    plt.plot(X, series[3],label="nltk")
    plt.plot(X, series[4],label="all")
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel('Sentence length')
    plt.ylabel('Count')
    plt.show()
    
def plot_mean_grade_distribution(modified_train_df):
    sns.histplot(modified_train_df.meanGrade, kde=True, color = 'grey')
    plt.show()
    
def plot_number_characters(modified_train_df):
    modified_train_df.edited.str.len().hist(color='grey')
    plt.show()
    
def plot_number_words(modified_train_df):
    modified_train_df.edited.str.split().\
    map(lambda x: len(x)).\
    hist(color='grey', grid=False)
    plt.ylabel('Count')
    plt.xlabel('Words per sentence')
    plt.show()
    
def plot_stop_words(dic):
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:15] 
    x,y=zip(*top)
    plt.barh(x,y, color = 'grey')
    plt.xlabel('Count')
    plt.ylabel('Stop words')
    plt.show()
    
def plot_top_ngrams(top_n_bigrams):
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x, color='grey')
    plt.xlabel('Count')
    plt.ylabel('Bigrams')
    plt.show()