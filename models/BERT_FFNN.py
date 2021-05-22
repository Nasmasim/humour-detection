import torch.nn as nn
import torch

from transformers import BertModel

# Bert download
bert_model = BertModel.from_pretrained('bert-base-uncased')

#@title Bert-FFNN (FFNN)
class FFNN(nn.Module):
    def __init__(self, hidden_dim=[128, 64, 32], embedding_dim=768):  
        super(FFNN, self).__init__()
        
        self.embed_dim = embedding_dim
        hidden_dim_all = [self.embed_dim] + hidden_dim
        # self.drop_out = nn.Dropout(0.2)
        # hidden layer
        blocks = []
        for i in range(len(hidden_dim_all)-1):
          blocks.append(nn.Sequential(
              nn.Linear(hidden_dim_all[i], hidden_dim_all[i+1], bias=True),
              nn.LeakyReLU()
              ))
        self.ffnn = nn.Sequential(*blocks)
        
        # output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim[-1], 1),
            nn.ReLU())  
        
    def forward(self, bert_id, bert_attn):
        embedded = bert_model(input_ids=bert_id, attention_mask=bert_attn)[0]
        
        # We want to use one vector (len=768) to represent one sentence
        ffnn_input, _ = torch.max(embedded, dim=1, keepdim=False)
        ffnn_output = self.ffnn(ffnn_input)
        out = self.output(ffnn_output)
        
        return out