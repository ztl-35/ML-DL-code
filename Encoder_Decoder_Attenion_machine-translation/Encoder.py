import torch
import torch.nn as nn
from torch.autograd import Variable

BATCH_SIZE = 64
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    
class Encoder(nn.Module):
    # input_size 代表这一次输入多少个词
    # hidden_dim 代表隐层多大
    def __init__(self, vocub_size, hidden_dim, n_layers=1, dropout=0.1, is_directional=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocub_size, hidden_dim)
        self.layers = n_layers
        self.dropout = dropout

        if is_directional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        # x0, h0/不用输入c0    input = sequence * batch * input_embedded(虽然定义了batch_first=true，但是
        # 在网络中输入需要按照规定的格式，只是网络在识别的时候，会把第二维度识别成)
        self.GRU = nn.GRU(hidden_dim, hidden_dim,
                          num_layers=self.layers, dropout=self.dropout, bidirectional=is_directional)

    def forward(self, input_word_index, input_real_length, encoder_init_hidden=None):
        # 64 * 9 * 256
        input_word_index = Variable(input_word_index)
        input_real_length = torch.Tensor(input_real_length)
        embedded = self.embedding(input_word_index)

        # 压缩padding状态
        input_packed = nn.utils.rnn.pack_padded_sequence(embedded, input_real_length)
        encoder_packed, encoder_last_hidden = self.GRU(input_packed, encoder_init_hidden)

        # 解压状态
        encoder_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_packed)
        # 双向gru  最后一层每个时间步输出的是 h = 【h正向，h逆向】 所以最后需要正向和逆向的结果分开处理（这里是直接相加），保证了隐层输出维度的一致性
        # encoder_packed = seq * batch * (hidden_size * num_directions)   [batch first = False]

        if self.num_direction == 2:
            encoder_pad = encoder_pad[:, :, :self.hidden_dim] + encoder_pad[:, :, self.hidden_dim:]
        else:
            pass

        return encoder_pad, encoder_last_hidden

    def initHidden(self):
        # 注意: batch_first = True 的时候，这个初始化或者默认初始化的隐层还是要按照原来的size设置（sequence * batch * word_emb）
        return torch.randn(self.num_direction, BATCH_SIZE, self.hidden_dim, device=device)
