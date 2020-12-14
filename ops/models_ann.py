# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.temporal_modeling import *
from torchvision.models._utils import IntermediateLayerGetter as MidLayerGetter
from torchsummary import summary


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False, concat = "", extra_temporal_modeling = False, prune_list = [], is_prune = ""):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.num_class = num_class

        self.concat = concat
        self.extra_temporal_modeling = extra_temporal_modeling
        self.is_prune = is_prune
        self.prune_list = prune_list
        
        #self.activate = nn.Tanh()      #if use cross entropy, output value can't be nagative //log DNE
        self.activate = nn.Sigmoid()

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality in ["RGB", "Depth"] else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'RGB' and self.new_length == 2:                 #this condition means data fusion
            print("Converting the ImageNet model to a RGB-depth fusion init model")
            self.base_model = self._construct_fuse_model(self.base_model)
            print("Done. RGB-depth fusion model ready...")
        elif self.modality == 'Depth':
            print("Converting the ImageNet model to a depth init model")
            self.base_model = self._construct_depth_model(self.base_model)
            print("Done. Depth model ready...")
        elif self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        if consensus_type == 'conv1d':
            self.consensus = nn.Conv1d(in_channels = num_segments, out_channels = 1, kernel_size = 1, bias = False)
            self.consensus2 = nn.Conv1d(in_channels = num_segments*2, out_channels = 1, kernel_size = 1, bias = False)
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if 'efficientnet' in self.base_model_name:
            setattr(self.base_model, self.base_model.dropout, nn.Dropout(p=self.dropout))
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        elif self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            from archs.Resnet_ann_v2 import __resnet__
            self.base_model = __resnet__(base_model, True if self.pretrain == 'imagenet' else False, self.extra_temporal_modeling, 
                                                self.num_segments, self.num_class)
            if self.is_shift:
                #summary(self.base_model,(3,256,256))
                #print(self.base_model)
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments, n_div=self.shift_div, place=self.shift_place, 
                                    temporal_pool=self.temporal_pool, concat = self.concat, prune_list = self.prune_list[0] if self.is_prune in ['input', 'inout'] else {}, 
                                    prune = True if self.is_prune in ['input', 'inout'] else False)
            if self.is_prune in ['output', 'inout']:
                print('prune from conv1 and conv2...')
                from ops.prune import make_prune_conv
                make_prune_conv(self.base_model, place=self.shift_place)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'



            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif 'efficientnet' in base_model:
            from efficientnet_pytorch import EfficientNet
            self.base_model = EfficientNet.from_pretrained(base_model, temporal_modeling = self.extra_temporal_modeling, 
                                                            segment = self.num_segments, num_class = self.num_class)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_efficientnet_shift
                make_efficientnet_shift(self.base_model, self.num_segments, n_div=self.shift_div,
                                     place=self.shift_place, temporal_pool=self.temporal_pool, concat = self.concat)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.dropout = '_dropout'
            self.base_model.last_layer_name = '_fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m_name, m in self.base_model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    if 'extra_layer_batchnorm' in m_name and self.is_prune == '':
                        continue
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, concat):
        first_conv_weight = []
        first_conv_bias = []

        extra_weight = []
        extra_bias = []
        extra_bn = []

        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        embedding_weight = []
        gru_weight = []

        conv_cnt = 0
        bn_cnt = 0
        for m_name, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                elif 'extra_layer_' in m_name:
                    extra_weight.append(ps[0])
                    if len(ps) == 2:
                        extra_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if 'extra_layer_fc' in m_name:
                    extra_weight.append(ps[0])
                elif self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if 'extra_layer_fc' in m_name:
                        extra_bias.append(ps[1])
                    elif self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                extra_bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
                elif self.is_prune == '' and 'extra_layer_batchnorm' in m_name:
                    extra_bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.LSTM):
                extra_weight.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.modules.sparse.Embedding):
                ps = list(m.parameters())
                #print(m_name)
                #print(ps)
                for p in ps:

                    embedding_weight.append(p)
            elif isinstance(m, torch.nn.modules.rnn.GRU):
                ps = list(m.parameters())
                #print(m_name)
                for p in ps:
                    #print(ps)

                    gru_weight.append(p)

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality in ['Flow', 'Depth'] else 1 if self.new_length == 2 else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality in ['Flow', 'Depth'] else 2 if self.new_length == 2 else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},

            # for concat setting
            {'params': extra_weight, 'lr_mult': 10 if self.is_prune == '' else 1, 'decay_mult': 1,
             'name': "extra_weight"},
            {'params': extra_bias, 'lr_mult': 20 if self.is_prune == '' else 2, 'decay_mult': 0,
             'name': "extra_bias"},
            {'params': extra_bn, 'lr_mult': 10 if self.is_prune == '' else 1, 'decay_mult': 0,
             'name': "extra_bn"},

            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},


            {"params": embedding_weight, "lr_mult":5, "decay_mult":1,
            "name":"embedding_weight"},
            {"params": gru_weight, "lr_mult":5, "decay_mult":1,
            "name":"gru_weight"},
        ]
    def trainEncoder(self, input_tensor):
        pass

    def forward(self, input, input_caption, no_reshape=False):
        #print("in model forward")
        #print("input size=",input.size())
        #print(input_caption)
        #print("-"*50)

        if not no_reshape:
            if self.modality == "RGB":
                #print("models 357")
                sample_len = (3 * self.new_length) if self.new_length == 1 else 4
            elif self.modality == "Depth":
                #print("models 360")
                sample_len = 1 * self.new_length
            elif self.modality == "Flow":
                #print("models 363")
                sample_len = 2 * self.new_length
            #sample_len = (3 if self.modality in ["RGB", "Depth"] else 2) * self.new_length
            elif self.modality == 'RGBDiff':
                #print("models 367")
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            if self.extra_temporal_modeling:
                #print("models 371")

                base_out, spatial_feature = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))   # base_out -> (Batch_size * segment, 2048)

                #print("base_out size",base_out.size())
                #print("spatial_freature=",spatial_feature.size())
            else:
                #print("models 375")
                #print("input size",input.size())
                #print(input.view((-1, sample_len) + input.size()[-2:]))
                #print((-1, sample_len) + input.size()[-2:])
                
                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]),input_caption.view(-1, input_caption.size()[2],1))
                #print("base out size",base_out.size())
                base_out = base_out.squeeze()
                #print(base_out)
            
        else:
            #print("models 379")
            base_out = self.base_model(input, input_caption)
            #print("base out size",base_out.size())
        if self.dropout > 0 and 'efficientnet' not in self.base_model_name:
            pass
            #print("models 383")
            #base_out = self.new_fc(base_out)
            #print("base out size",base_out.size())

        if not self.before_softmax:
            #print("models 387")
            base_out = self.softmax(base_out)
        
        output = 0
        if self.reshape:
            #print("models 392")
            if self.is_shift and self.temporal_pool:
                #print("models 394")
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                output = self.consensus(base_out)

                if self.extra_temporal_modeling != "":
                    #print("models 399")
                    return output.squeeze(1) + spatial_feature

            elif self.extra_temporal_modeling:
                #print("models 403")
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                spatial_feature = spatial_feature.view((-1, self.num_segments) + spatial_feature.size()[1:])

                base_out = torch.cat((base_out, spatial_feature), dim=1)
                output = self.consensus2(base_out)
                spatial_feature = self.consensus(spatial_feature)
                
                return output.squeeze(1), spatial_feature.squeeze(1)
            else:
                #print("465")
                #print(base_out.size())
       
                #output = self.activate(base_out)
                #print(output)
                #print("outputsize-----",output.size())
                return base_out.view(1,base_out.size()[0],base_out.size()[1])
                ###return base_out.view(1,base_out.size()[0], base_out.size()[1])
                #return base_out.squeeze(1).unsqueeze(0)
                #return base_out.squeeze(1)
                #print("models 413")
                #print("base_out size=",base_out.size())
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])     #(batch_size, num_segment, class_num)
                #print("base_out size=",base_out.size())
                base_out = base_out.view((-1) + base_out.size()[1:])
                #print("base_out size=",base_out.size())
                output = self.consensus(base_out)
                #print("output size", output.size())
            #print("models 416")
            #print("outputsize+++++",output.size())
            #return output.squeeze(1)
            return output.unsqueeze(0)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_fuse_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (4, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return base_model

    def _construct_depth_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (1 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return base_model

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality in ['RGB', 'Depth']:
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False),
                                                       ])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66, .625]),
                                                       ])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

