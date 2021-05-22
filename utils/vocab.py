import codecs
import numpy as np

#@title Default create vocab (create_covab)
def create_vocab(data):
    """
    Creating a corpus of all the tokens used
    """
    # Let us put the tokenized corpus in a list
    tokenized_corpus = [] 
    for sentence in data:
        tokenized_sentence = []
        for token in sentence.split(' '): # simplest split is
            tokenized_sentence.append(token)
        tokenized_corpus.append(tokenized_sentence)

    # Create single list of all vocabulary
    # Let us put all the tokens (mostly words) appearing in the vocabulary in a list
    vocabulary = [] 
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus

def get_word2idx(file_path, joint_vocab):
    # We create representations for our tokens
    wvecs = [] # word vectors
    word2idx = [] # word2index
    idx2word = []
    
    # This is a large file, it will take a while to load in the memory!
    with codecs.open(file_path, 'r','utf-8') as f:
      index = 1
      for line in f.readlines():
        # Ignore the first line - first line typically contains vocab, dimensionality
        if len(line.strip().split()) > 3:
          word = line.strip().split()[0]
          if word in joint_vocab:
              (word, vec) = (word,
                         list(map(float,line.strip().split()[1:])))
              wvecs.append(vec)
              word2idx.append((word, index))
              idx2word.append((index, word))
              index += 1
    
    wvecs = np.array(wvecs)
    word2idx = dict(word2idx)
    idx2word = dict(idx2word)
    return wvecs, word2idx, idx2word