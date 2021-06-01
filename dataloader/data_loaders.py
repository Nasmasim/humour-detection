from __future__ import absolute_import

import torch
from torch.utils.data import Dataset, random_split
import numpy as np

from utils.vocab import create_vocab

#@title Dataset definition
class Task1Dataset(Dataset):

    def __init__(self, train_data, labels):
        self.x_train = train_data
        self.y_train = labels

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]    

#@title Convert corpus to Bert id and attention masks (get_input_bert)
def get_input_bert(df, tokenizer, col='edited'):
    """
    Convert corpus to ids (indices) and attention masks
    """
    # Find the maximum length of all sentences in the corpus 
    word_list, corpus = create_vocab(df[col])
    pad_len = max([len(sent) for sent in corpus])
    # Get BERT embeddings for all words in our corpus
    word_list_unique = list(set(word_list))
    
    input_id_mask = []
    for _, row in df.iterrows():
      # encoder_plus returns a dictionary of 'input_ids', 'token_type_ids', 'attention_mask', etc
      bert_enc = tokenizer.encode_plus(row[col], add_special_tokens=True, 
                                      truncation='longest_first', 
                                      max_length=pad_len)
      padding_len = pad_len - len(bert_enc['input_ids'])
      mask = [1] * len(bert_enc['input_ids']) 

      # Pad each sentence to have the same length by adding zeros to the end 
      padded = bert_enc['input_ids'] + ([tokenizer.pad_token_id] * padding_len)
      # Tell BERT to ignore padded words in each sentence 
      mask += ([0] * padding_len)
      input_tensor = torch.from_numpy(np.asarray(padded))
      mask_tensor = torch.from_numpy(np.asarray(mask))
      input_id_mask.append((input_tensor, mask_tensor))

    return input_id_mask

# Baseline dataloader defined by joining back train and dev and splitting randomly
def get_dataloaders(input_data, 
                    targets, 
                    train_split, batch_size):
    """
    Using outputs from 'get_input_bert', create dataloaders for training. Make 
    random splits with the training dataset. 
    """
    train_and_dev = Task1Dataset(input_data, targets)
    train_examples = round(len(train_and_dev) * train_split)
    dev_examples = len(train_and_dev) - train_examples
    # split datasets
    train_dataset, dev_dataset = random_split(train_and_dev,
                                              (train_examples,
                                                dev_examples))
    # load into torch
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                              batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                              shuffle=False, 
                                              batch_size=batch_size)

    return train_loader, dev_loader

# Data loaders for both train and validation dataset (dev or test)
def get_dataloaders_no_random_split(input_data_train, 
                                    targets_train, 
                                    input_data_valid,
                                    targets_valid, 
                                    batch_size):
    """
    Using outputs from 'get_input_bert', create dataloaders for training.
    """
    train_ds = Task1Dataset(input_data_train, targets_train)
    valid_ds = Task1Dataset(input_data_valid, targets_valid)
    # load into torch
    train_loader = torch.utils.data.DataLoader(train_ds, 
                                                shuffle=True, 
                                                batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(valid_ds, 
                                              shuffle=False,
                                              batch_size=batch_size)

    return train_loader, dev_loader

#@title Padding mini-batches (collate_fn_padd)
def collate_fn_padd(batch):
    '''
    We add padding to our minibatches and create tensors for our model
    '''

    batch_labels = [l for f, l in batch]
    batch_features = [f for f, l in batch]

    batch_features_len = [len(f) for f, l in batch]

    seq_tensor = torch.zeros((len(batch), max(batch_features_len))).long()

    for idx, (seq, seqlen) in enumerate(zip(batch_features, batch_features_len)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    batch_labels = torch.FloatTensor(batch_labels)

    return seq_tensor, batch_labels
