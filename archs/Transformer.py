from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from ops.temporal_modeling import *

import unicodedata
import string
import re
import random
import time
import pickle



def load_object(filename):
    with open(filename,"rb") as dic:
        obj = pickle.load(dic)
        return obj

class TransformerModel(nn.Module):
    def __init__(self, ntoken=None, ninp=128, nhead=8, nhid=2048, nlayers=8, dropout=0.5):
        
        #ntoken: len(dictionary)
        #ninp : embedding dimension
        #nhead: # of multiheadattention 
        #nhid : dim(feedforward network model)
        #nlayer: # of nn.TransofrmerEncoderLayer
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoder, TransformerDecoderLayer
        if not ntoken:
            self.ntoken = len(load_object("engDictAnn.pkl"))+2
        else:
            self.ntoken = ntoken
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encdoer_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encdoer_layers, 2)
        #self.encoder = nn.Embedding(self.ntoken, ninp)
        self.encoder = nn.Linear(2048, ninp)
        self.ninp = ninp
        #self.transformer_decoder = Transformer(d_model=2048, nhead=nhead, )

        self.embedded = nn.Embedding(self.ntoken, ninp)
        
        self.decoder = nn.Linear(ninp, self.ntoken)
        
        decoder_layers = TransformerDecoderLayer(d_model=ninp, nhead=8, )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=4)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        #print("in  generate_sqaure_subsequent_maks")
        #print(sz)
        mask = (torch.triu(torch.ones(sz, sz)) == 1 ).transpose(0,1)
        
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None, tar=None):
        #mem = self.encoder(src)
        src = self.encoder(src) * (self.ninp)**2
        #src = self.pos_encoder(src)
        mem = self.transformer_encoder(src)
        print("src size=",src.size())
        
        #print("mem size=",mem.size())
        #output = self.transformer_decoder(src, tar)
        #mem = self.transformer_encoder(src)
        print("mem size=",mem.size())
        #print("tar=",tar.size())
        #print("mask=",mask.size())
        #print(torch.zeros((1,1), dtype=torch.long).cuda().size())
        #tar = torch.cat((torch.zeros((1,1), dtype=torch.long).cuda(),tar), dim = 1)
        #print("tar size()", tar.size())
        if mask is None: #testing
            _tar = torch.zeros((1,1), dtype=torch.long).cuda()
            output = torch.tensor([]).cuda()
            word = 0
            
            while output.size(0) < 64 and word !=1:
                _tar = self.embedded(_tar.T)
                _tar = self.pos_encoder(_tar)
                #print("_tar size()", _tar.size())
                output = self.transformer_decoder(_tar, mem)

                output = self.decoder(output)
                words = output.argmax(-1).T
                word = words[0,-1].item()

                _tar = torch.cat((words, torch.zeros((1,1), dtype=torch.long).cuda()),dim=1)
                #print("_tar size", _tar.size())
                #print(words)
                
            print("word=",word)


        
        else:           #training
            tar = torch.cat((torch.zeros((1,1), dtype=torch.long).cuda(), tar[...,:-1]), dim=1)
            tar = self.embedded(tar.T)
            tar = self.pos_encoder(tar)
            #print("tar size()", tar.size())
            output = self.transformer_decoder(tar, mem, mask)
            
            output = self.decoder(output)
            #print(output)
            #print(output.size())
            #output = self.decoder(src, mem)
            #output = F.log_softmax(output, dim=1)
        print(output.max(dim=2))
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        import math
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)