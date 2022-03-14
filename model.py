import torch
import torch.nn as nn
import math
from modules.pvt_str import create_pvtstr

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.pvtstr= create_pvtstr(num_tokens=opt.num_class, model=opt.TransformerModel)
        return

    def forward(self, input, text, is_train=True, seqlen=25):
            prediction = self.pvtstr(input, seqlen=seqlen)
            return prediction
