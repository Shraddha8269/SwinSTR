import torch
import torch.nn as nn
import math
from modules.swinstr2 import create_swinstr

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.swinstr= create_swinstr(num_tokens=opt.num_class, model=opt.TransformerModel)
        return

    def forward(self, input, text, is_train=True, seqlen=25):
            prediction = self.swinstr(input, seqlen=seqlen)
            return prediction
