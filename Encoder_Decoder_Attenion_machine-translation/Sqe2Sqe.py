import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from Attention import Attention
from torch.autograd import Variable

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
else:
    pass

class Sqe2Sqe(nn.Module):
    def __init__(self, encoder, decoder):
        super(Sqe2Sqe, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = sqe_len * batch_size
        # trg = len * batch_size
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocub_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocub_size).to(device)

        # src : ? (Sqe 和 sqe_len )
        encoder_output, hidden = self.encoder(src)
        # 取hidden最后几层的
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attention_weight = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda("0")
        return outputs

