import torch
import torch.nn as nn
import math
from modules.uni_str import create_unistr

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.unistr= create_unistr(num_tokens=opt.num_class, model=opt.TransformerModel)
        return

    def forward(self, input, text, is_train=True, seqlen=25):
            prediction = self.unistr(input, seqlen=seqlen)
            return prediction
