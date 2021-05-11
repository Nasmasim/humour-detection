#################
#
# PREPROCESSING
#
#################

import copy 
import re

# Definining a function to get the complete edited version of the headlines
def create_edited_sentences(df):
    """
    - Add a column in df called edited that contains edited headlines
    params:
      df: dataframe with columns "original" and "edit"
    output:
      edited_df: a copy of modified df where edit it the new headline
    """
    edited_lst = []
    edited_df = copy.copy(df)
    for idx, row in df.iterrows():
      # Replace everything between and including "<" and "/>" with the word in col edit
      edited_sentence = re.sub("<.*?/>", row['edit'], row['original'])
      edited_lst.append(edited_sentence)
    edited_df["edited"] = edited_lst
    return edited_df

#Checking special characters in headlines
def check_non_alpha_char(df,column='edited'):
  char_dict = {}
  for edited_headline in df[column]:
    for present_char in edited_headline:
      if not present_char.isalpha():
        if present_char in char_dict:
          char_dict[present_char] = char_dict[present_char]+1
        else:
          char_dict[present_char] = 1
  print(char_dict)

# Sentence preprocessing on strings before tokenization
def question_sentence_preprocessing(s):
    """
    - Change "'t" to "not"
    - Remove punctuations except question mark (?)
    - Remove trailing white spaces 
    params:
      s: original sentence
    output:
      s: edited sentence
    """
    s = s.lower()
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r'\1', s)
    s = re.sub(r'[^\w\s\?]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def full_sentence_preprocessing(s):
    """
    - Remove all special characters
    - Remove trailing white spaces
    """
    s = s.lower()
    # remove all special characters using translate
    remove = "[!”#$%&’'‘()*+,-./:;<=>?@[\]^_`{|}~]:"
    removed = str.maketrans("", "", remove)
    s = s.translate(removed)
    s = re.sub(r'[^\w\s\?]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def create_custom_vocab(data,stopwords=[]):
    """
    - Remove stop words in data
    - Tokenize sentences in data
    - Create a vocabulary (list) and tokenized corpus (nested list)
    params:
      - data: column of a df that contains our corpus
      - stopwords: list of words to remove 
    output:
      - vocabulary: list of words in the corpus 
      - tokenized_corpus: nested list containing all sentences broken apart 
      - word2idx: dictionary with unique words in the corpus as keys and 
                  indices as items 
    """
    #Based on the given create_vocab function
    tokenized_corpus = [] # Let us put the tokenized corpus in a list
    for sentence in data:
        tokenized_sentence = []        
        for token in sentence.split(' '): # simplest split is
            if token not in stopwords:
              tokenized_sentence.append(token)
        tokenized_corpus.append(tokenized_sentence)
    # Create single list of all vocabulary
    vocabulary = []  # Let us put all the tokens (mostly words) appearing in the vocabulary in a list
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                  vocabulary.append(token)
    word2idx = {w: idx+1 for (idx, w) in enumerate(vocabulary)}
    # reserve the firtst index (0) for the padding token
    word2idx['<pad>'] = 0

    return vocabulary, tokenized_corpus, word2idx

# Applying both question and full version to any dataframe and dropping useless values
def dataset_question_full_processing(df):
  df['question_edited'] = df['edited'].map(lambda x : question_sentence_preprocessing(x))
  df['full_edited'] = df['edited'].map(lambda x : full_sentence_preprocessing(x))
  df = df.drop(['original','edit'],axis=1)
  if 'grades' in df.columns:
    df = df.drop(['grades'],axis=1)
  return df