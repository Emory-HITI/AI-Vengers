import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod
import timm

class DenseNet(nn.Module):
    def __init__(self, use_pretrained, num_extra=0, n_outputs = 2):
        super().__init__()
        self.num_extra = num_extra
        self.model_conv = mod.densenet121(pretrained= use_pretrained)         
        self.num_ftrs = self.model_conv.classifier.in_features + num_extra
        self.model_conv.classifier = nn.Identity()
        self.class_conf =  nn.Linear(self.num_ftrs,n_outputs)

    def forward(self,x,*args):
        assert(len(args) <= 1) 
        img_conv_out = self.model_conv(x)
        if self.num_extra:
            assert(args[0].shape[1] == self.num_extra)        
            img_conv_out = torch.cat((img_conv_out, args[0]), -1)
        out = self.class_conf(img_conv_out)
        return out
       
class VisionTransformer(nn.Module):
    def __init__(self, use_pretrained, num_extra=0, n_outputs = 2):
        super().__init__()
        self.num_extra = num_extra
        self.model_conv = timm.create_model('vit_deit_small_patch16_224', pretrained= use_pretrained)         
        self.num_ftrs = self.model_conv.head.in_features
        self.model_conv.head = nn.Identity()
        self.class_conf = nn.Linear(self.num_ftrs, n_outputs)
    
    def forward(self,x,*args):
        assert(len(args) <= 1) 
        img_conv_out = self.model_conv(x)
        if self.num_extra:
            assert(args[0].shape[1] == self.num_extra)        
            img_conv_out = torch.cat((img_conv_out, args[0]), -1)
        out = self.class_conf(img_conv_out)
        return out