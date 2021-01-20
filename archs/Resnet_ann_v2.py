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

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

TFDEM = False
SEGMENT = 8
CLASSES = 101


def load_object(filename):
    with open(filename,"rb") as dic:
        obj = pickle.load(dic)
        return obj
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        print("in EncoderRNN")
        embedded = self.embedding(input).view(1, 1, -1)
        print(embedded.size())
        output = embedded
        print(output.size())
        output, hidden = self.gru(output, hidden)
        print(output.size(),hidden.size())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,).cuda()

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros((1, 1, self.hidden_size),dtype=torch.float).cuda()

class FeatureEncoder(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, input, hidden, ):
        pass

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size=128, output_size=None, dropout_p=0.1, feature_length=64): #64 = input word maximum length, 128 = embedded dim
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word2index = load_object("engDictAnn.pkl")
        self.output_size = len(self.word2index)+2 #+sos eos
        
        self.dropout_p = dropout_p
        self.feature_length = feature_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.feature_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output):
        repeat_encoder_output = encoder_output
        #repeat_encoder_output = encoder_output.repeat(self.feature_length,1)
        #print(repeat_encoder_output[0,0:5], repeat_encoder_output[1,0:5])
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        #print("embedded size=",embedded.size())
        #print("hidden size=", hidden.size())
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]),1)),  dim=1
        )
        #print("attn_weights size = {}, encoder_output.size()={}, repeat_encoder_output={}".format(attn_weights.size(), encoder_output.size(), repeat_encoder_output.size()))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), repeat_encoder_output)
        #print("attn_applied=",attn_applied.size())
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        #print("214",output.size())
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #print("217",output.size())
        output = F.log_softmax(self.out(output[0]), dim=1)
        #print("219",output.size())
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size).cuda()

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp=128, nhead=8, nhid=128, nlayers=8, dropout=0.5):
        #ntoken: len(dictionary)
        #ninp : embedding dimension
        #nhead: # of multiheadattention 
        #nhid : dim(feedforward network model)
        #nlayer: # of nn.TransofrmerEncoderLayer
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        if not ntoken:
            self.ntoken = len(load_object("engDictAnn.pkl"))
        else:
            self.ntoken = ntoken
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encdoer_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encdoer_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, self.ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1 ).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0,0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, decoder=None, encoder=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lastconv = nn.Conv1d(512 * block.expansion, 1, kernel_size=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Linear(512 * block.expansion, 1)
        #self.decoder = decoder(256,100) #if add, model optim paramter needs to add
        #self.encoder = encoder(64,256)
        #self.input_size = 64
        self.hidden_size = 128
        self.word2index = load_object("engDictAnn.pkl")
        self.input_size = len(self.word2index)+2 #+sos eos
        self.embedding = nn.Embedding(self.input_size, self.hidden_size) #prevent errors
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        self.gru4feature = nn.GRU(2048, 128)
        self.decoder = decoder(self.hidden_size, self.input_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        if TFDEM:
            self.extra_layer0 = TemporalModeling(num_segment = SEGMENT, num_class = CLASSES, backbone = 'ResNet')
    
    def initHidden(self):
        return torch.zeros((1, 1, self.hidden_size),dtype=torch.float).cuda()
    def encoder(self,input, hidden ):
        #print("in encoder")
        #print(input)
        embedded = self.embedding(input).view(1, 1, -1)
        #print(embedded)
        #print(embedded.size())
        output = embedded
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        #print("output.size={}, hidden.size={}".format(output.size(),hidden.size()))
        #print(embedded)
        #print(output)
        return output, hidden


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,y):
        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size).cuda()
        #encoder_hidden = self.encoder.initHidden()
        encoder_hidden = self.initHidden()
        #print("encoderf hidden size",encoder_hidden.size())
        #print(y.size())
        hidden_state = torch.zeros((y.size()[0],self.hidden_size),dtype=torch.float).cuda()
        #EOS = torch.ones(1, dtype=torch.long).cpu()
        #print("+"*50)
        #print(y.size())

        #print(y)
        for i in range(y.size()[0]):
            #print("cap_num.data*********{}".format(cap_num.data))

            for j in range(y.size()[1]):
                #print("i={},j={}".format(i,j))
                #print(y)
                #print(y[i,j])
                #print("{0:*^50}".format("encoder hidden"))
                #a = i.data
                #a = a.cpu()
                #print(a)
                #print(i)
                #print(encoder_hidden)
                #encoder_outputs, encoder_hidden = self.EncoderRNN(ic, encoder_hidden)
                tmp , encoder_hidden = self.encoder(y[i,j], encoder_hidden)
                #print(encoder_hidden)
                #print("encoder finish\t", encoder_hidden.size(),"\t",tmp.size())
                if y[i,j] == 1: #EOS
                    #print("eos")
                    break
                #print("eos test pass")
            #print(encoder_hidden.view(1,-1))
            #torch.cat((hidden_state, encoder_hidden.view(1,-1).float().cuda()),0)
            hidden_state[i] = encoder_hidden.view(1,-1)
            #print("hidden state size=",hidden_state.size())
            #print(hidden_state)
            encoder_hidden = self.initHidden()
        #print("hidden state size=",hidden_state.size())
        #print("resnet")
        #print("x.size",x.size())
        x = self.conv1(x)
        #print("conv1 x.size",x.size())
        x = self.bn1(x)
        #print("bn1 x.size",x.size())
        x = self.relu(x)
        #print("relu x.size",x.size())
        x = self.maxpool(x)
        #print("maxpool x.size",x.size())
        if TFDEM:
            spatial_temporal_feature = self.extra_layer0(x)
        hidden_state = hidden_state.view(x.size()[0],1,x.size()[2],-1)

        # concatenate caption feature with video feature
        #print("encoder hidden size=",hidden_state.size())
        #cat = torch.zeros((x.size()[0],x.size()[1],x.size()[2],x.size()[3]+hidden_state.size()[-1]), dtype=torch.float).cuda()
        cat = hidden_state.repeat(1,64,1,1).cuda()
        #print("after cat repeat size=", cat.size())
        #cat_var = torch.autograd.Variable(cat)
        cat_var = torch.cat((x,cat),3).cuda()
        #for frame in range(x.size()[0]):
        #    for channel in range(x.size()[1]):
        #        for w in range(x.size()[2]):

                    #cat_var[frame, channel, w,:x.size()[3]] = x[frame, channel, w]
                    #cat_var[frame, channel, w,x.size()[3]:] = hidden_state[frame, 0, w]

        x = torch.autograd.Variable(cat_var).cuda()
        #print("x size = ",x.size())
        x = self.layer1(x)
        #print("layer1 x.size",x.size())
        x = self.layer2(x)
        #print("layer2 x.size",x.size())
        x = self.layer3(x)
        #print("layer3 x.size",x.size())

        x = self.layer4(x)
        #print("layer4 x.size",x.size())

        x = self.avgpool(x)
        #print("avgpol x.size",x.size())
        ###x = torch.flatten(x, 1)
        #print(x.size())
        #x = x.unsqueeze(-1)
        #print(x.size())
        #####x = self.lastconv(x)
        ##x, hidden = self.decoder(x)
        #print("*"*50)
        #print(x.size())
        #x = self.fc(x)
        #x, hidden = self.decoder(x)
        #print("fc ", x.size())
        hidden_state = self.initHidden()
        encoder_output = torch.zeros((64, 128), dtype=torch.float).cuda()
        for idx, frame in enumerate(x):
            
            #print("frame feature dim=", frame.size())
            encoder_output[idx], hidden_state = self.gru4feature(frame.view(1,1,-1), hidden_state)
            #print("encoder_output size=",encoder_output.size())
        #exit()
        #print("hidden state size = ",hidden_state.size())
        if TFDEM:
            return x, spatial_temporal_feature
        else:
            return encoder_output


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs, decoder=AttnDecoderRNN, encoder=EncoderRNN)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        model_dict = model.state_dict()
        replace_dict = []
        for k, v in state_dict.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in state_dict and k.replace('.net', '') in state_dict:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            state_dict[k_new] = state_dict.pop(k)
        keys1 = set(list(state_dict.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    print("in resnet 312")
    print(model)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def __resnet__(arch, pretrained, temporal_modeling, segment, num_class):
    arch_dict = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
           'resnet152': resnet152, 'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d,
           'wide_resnet50_2': wide_resnet50_2, 'wide_resnet101_2': wide_resnet101_2}
    global TFDEM
    TFDEM = temporal_modeling
    global SEGMENT
    SEGMENT = segment
    global CLASSES
    CLASSES = num_class

    return arch_dict[arch](pretrained = pretrained)
