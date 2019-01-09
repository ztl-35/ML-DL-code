import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 10


class EncoderRnn(nn.Module):
    # 4489(训练的单词个数) 256（hidden 的大小）
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRnn, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 4489 * 256(每个词用256维度表示)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # input: x_dim = [seq, batch, x_dim]  h_dim = [层数*方向， batch, h_dim]
        # output: out = [seq, batch, h_dim*方向] h_t = [层数*方向， batch, h_dim]
        self.GRU = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # index
        input = input.unsqueeze(1)
        # 1 * 1=[[index]]
        # embedding 是输入的初始化好的索引，返回对应索引的词向量矩阵 [1, 1, 256]
        embedded = self.embedding(input)
        # [1, 1, 256]
        output = embedded.permute(1, 0, 2)

        # 1*1*256  1*1*256
        for i in range(self.n_layers):
            output, hidden = self.GRU(output, hidden)
        # output = n * hidden_size   hidden=1 * hidden_size(当前最后一步的网络状态输出)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    # 256  2923
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # 这里和EncoderRnn不一样了，直接定义多少个输出字的编码   2923*256
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        # 输出预测空间概率 256 -> 2923(给每个词一个概率，这样比较哪个大就输出哪个)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):

        # 返回词向量矩阵
        output = self.embedding(input) # batch, 1, hidden
        output = output.permute(1, 0, 2) # 1, batch, hidden
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.GRU(output, hidden)

        print('output[0]', output[0])

        output = self.softmax(self.out(output[0]))
        # 返回 softmax 输出结果，和最后一层hidden
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttentionDecoderRnn(nn.Module):
    # 256 2923 layer=2
    def __init__(self, hidden_size, output_size, n_layers=1,
                 dropout=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layers
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 将原始的词向量输入与编码的每一个隐层的输出作为计算attention的输入
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)
        self.GRU = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        :param input:  batch, 1  decoder_input(开始就一个索引，就是SOS_TOKEN)
        :param hidden: 1, batch, hidden 与初始化的encoder的隐层状态一样
        :param encoder_outputs: length, hidden
        :return:
        '''
        # input = [[SOS_TOKEN index]]
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # batch, hidden   把所有的向量格式压缩成 batch*hidden的矩阵规模
        # 1*256
        embedded = embedded.squeeze(1)

        # 将每一步的词嵌入和第一步的输出结果进行整合，得到第一步decoder翻译的结果，然后再传入到第二步中，继续进行翻译和计算attention
        # 输出的维度是预测的最大长度序列个数 -》变成一个概率大小的长度  batch * max_length
        # torch.cat = 1* 512
        # attention_weight = 1 * 10(当前的解码输入 和hidden（encoder上下文向量）联合之后的结果作为当前解码词计算注意力权重的结果)
        attention_weigths = F.softmax(self.attention(torch.cat((embedded, hidden[0]), 1)))
        # 1 * 10 * 256
        encoder_outputs = encoder_outputs.unsqueeze(0)

        # attention 计算的结果（最大长度词上每一个添加一个权重，然后把256维度的表示全部相加，得到一个权重分配的解码表示）
        attention_applied = torch.bmm(attention_weigths.unsqueeze(1), encoder_outputs)

        # 此时改变输入，将当前解码的预先词向量与权重分配的结合起来（将这个预先此向量与上下文向量做了attention） 1 *1 *512
        output = torch.cat((embedded, attention_applied.squeeze(1)), 1)

        # 1 * 1 * 256
        output = self.attention_combine(output).unsqueeze(0)

        for i in range(self.n_layer):
            output = F.relu(output)
            output, hidden = self.GRU(output, hidden)
        # 1 * 输出词表长度(2925)
        # hidden = 1 * 1 * 256
        output = F.log_softmax(self.out(output.squeeze(0)))

        return output, hidden, attention_weigths

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result