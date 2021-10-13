import  torch
from    torch import nn
from    torch.nn import functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

def FakeNewsClassifier(Module):
  def __init__(self, bert, args):
    super(dream_model, self).__init__()
    
    self.max_len = args.tokenizer_max_length
    self.hidden_size = args.hidden_size
    
    self.embedding = bert
    self.classifier = nn.Sequential(
       Linear(self.hidden_size, 1),
       LeakyReLU(0.01), #from equation 4
    )
    
  
  def forward(self, inp):
    """
    I don't know what kind of input we need.
    For now, I assume the input is [inp_tensor, msk_tensor, seg_tensor]
      - inp_tensor = (batch_size, num_tweets, max_length), is the token ids from the tweets. e.g. the tweet is = "the fox quick brown", the token ids = [1, 2, 3, 4, 0, 0...] #just toy example
        batch_size = number of user (?)
        num_tweets = num of tweet per user
        max_length = max length of the tweet.
      - msk_tensor = (batch_size, num_tweets, max_length) = mask token. 1 indicates that the token is present, while 0 is not. E.g., the tweet is "the fox quick brown", the max_len is 512. then, the msk_tensor = [1, 1, 1, 1, 0, ....]
      - seg_tensor = (batch_size, num_tweets, max_length) = indicate the segment where the sentence belongs. The value is 1 where the tokens are present (same as msk_tensor)
    """
    inp_tensor, msk_tensor, seg_tensor = input
    batch_size, num_tweets, _ = inp_tensor.shape
    
    inp_tensor = inp_tensor.view(-1, self.max_len)
    msk_tensor = msk_tensor.view(-1, self.max_len)
    seg_tensor = seg_tensor.view(-1, self.max_len)
    inputs_hiddens, inputs = self.embedding(inp_tensor, msk_tensor, seg_tensor)
    
    # CLS token
    inputs = inputs.view([-1, num_tweets, self.bert_hidden_dim])
    
    user_embedding = inputs.max(dim=1)
    
    probs = self.classifier(user_embedding)
    probs = F.sigmoid(probs)
    
    return probs
    
    
