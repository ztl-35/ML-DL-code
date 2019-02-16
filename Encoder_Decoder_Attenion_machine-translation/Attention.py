import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, decoder_pre_hidden, encoder_outputs, src_len=None):
        '''

        :param decoder_pre_hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)

        '''

        timestep = encoder_outputs.size(0)

        # 因为decoder序列中前一个隐层状态要和所有encoder的outputs状态进行得分计算
        # 所以相当于要将这个hidden复制encoder所有时间步长度的份数
        # 如果decoder_pre_hidden = (1*2, 64, 256)    h = 64 * (2*timestep) * 256  这里如果decoder是双向的，是否后续代码有错误？？？ 后面的维度无法拼接
        h = decoder_pre_hidden.repeat(timestep, 1, 1).transpose(0, 1)

        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attention_energy = self.score(h, encoder_outputs)

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask).unsqueeze(1).to(device)  # [B,1,T]
            attention_energy = attention_energy.masked_fill(mask, -1e18)

        # B * 1 *T
        return F.softmax(attention_energy).unsqueeze(1)

    def score(self, decoder_copy_hidden, encoder_outputs):
        '''

        :param decoder_copy_hidden: 64 * (2*timestep) * 256
        :param encoder_outputs:
        :return:
        '''
        # [B*T*2H]->[B*T*H]
        energy = F.tanh(self.attention(torch.cat([decoder_copy_hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)

        # encoder_outputs = [B*T*H]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]

        # B * 1 * T
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)



