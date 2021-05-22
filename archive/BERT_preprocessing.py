
# from utils_tokenize import create_vocab
# import torch
# import numpy as np

# #@title Convert corpus to Bert id and attention masks (get_input_bert)
# def get_input_bert(df, tokenizer, col='edited'):
#     """
#     Convert corpus to ids (indices) and attention masks
#     """
#     # Find the maximum length of all sentences in the corpus 
#     word_list, corpus = create_vocab(df[col])
#     pad_len = max([len(sent) for sent in corpus])
#     # Get BERT embeddings for all words in our corpus
#     word_list_unique = list(set(word_list))
    
#     input_id_mask = []
#     for _, row in df.iterrows():
#       # encoder_plus returns a dictionary of 'input_ids', 'token_type_ids', 'attention_mask', etc
#       bert_enc = tokenizer.encode_plus(row[col], add_special_tokens=True, 
#                                       truncation='longest_first', 
#                                       max_length=pad_len)
#       padding_len = pad_len - len(bert_enc['input_ids'])
#       mask = [1] * len(bert_enc['input_ids']) 

#       # Pad each sentence to have the same length by adding zeros to the end 
#       padded = bert_enc['input_ids'] + ([tokenizer.pad_token_id] * padding_len)
#       # Tell BERT to ignore padded words in each sentence 
#       mask += ([0] * padding_len)
#       input_tensor = torch.from_numpy(np.asarray(padded))
#       mask_tensor = torch.from_numpy(np.asarray(mask))
#       input_id_mask.append((input_tensor, mask_tensor))

#     return input_id_mask

