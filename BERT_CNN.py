#@title Bert-CNN (CNN)
class CNN(nn.Module):
  def __init__(self, out_channels, window_size=[5, 3], 
               dropout_prob=0.3, embedding_dim=768):
    super(CNN, self).__init__()
    
    # in_channels -- 1 text channel
    # out_channels -- the number of output channels
    # kernel_size is (window size x embedding dim)
    self.conv1 = nn.Conv2d(
      in_channels=1, out_channels=out_channels[0],
      kernel_size=(window_size[0], embedding_dim))
    
    self.conv2 = nn.Conv2d(
        in_channels=out_channels[0], 
        out_channels=out_channels[1],
        kernel_size=(window_size[1], 1)  
    )
    self.dropout = nn.Dropout(dropout_prob)

    # the output layer
    self.output = nn.Sequential(
        nn.Linear(out_channels[-1], 1),
        nn.ReLU()
    )
    
        
  def forward(self, bert_id, bert_attn): 
    embedded = bert_model(input_ids=bert_id, attention_mask=bert_attn)[0]
    # Add one channel in order to use conv2d
    embedded = embedded.unsqueeze(1)
    feature_maps = self.conv1(embedded)
    feature_maps = F.relu(feature_maps)

    feature_maps = self.conv2(feature_maps)
    feature_maps = F.relu(feature_maps)
    # We reduced the last dimension (embedding dim) to one
    feature_maps = feature_maps.squeeze(3)
    
    # Apply the max pooling layer
    pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])
    
    pooled = pooled.squeeze(2)
    dropped = self.dropout(pooled)
    scores = self.output(dropped)
    
    return scores