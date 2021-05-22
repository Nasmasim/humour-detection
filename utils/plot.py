import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentence_length_stopwords(X, series):
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
    modified_train_df.meanGrade.hist( color = 'grey', bins=30, grid =False)
    plt.xlabel('Mean Grade')
    plt.ylabel('Count')
    
def plot_number_characters(modified_train_df):
    modified_train_df.edited.str.len().hist(color='grey', grid=False)
    plt.xlabel('Characters per sentence')
    plt.ylabel('Count')
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
    
def plot_loss_vs_epochs(question_train_losses_ff, question_valid_losses_ff, 
                        full_train_losses_ff,full_valid_losses_ff ):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)
    axes[0].plot(question_train_losses_ff, label="Question preprocessing training RMSE")
    axes[0].plot(question_valid_losses_ff, label="Question preprocessing validation RMSE")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    
    
    axes[1].plot(full_train_losses_ff, label="Full preprocessing training RMSE")
    axes[1].plot(full_valid_losses_ff, label="Full preprocessing validation RMSE")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    plt.suptitle("Loss Curves: BERT with FFNN")
    
    plt.savefig("FFNN_loss.png")
    plt.show() 