import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocub_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = vocub_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocub_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, vocub_size)

    def forward(self, input, last_hidden, encoder_outputs):
        '''
                :param word_input:
                    word input for current time step, in shape (B)
                :param last_hidden:
                    last hidden stat of the decoder, in shape (layers*direction*B*H)
                :param encoder_outputs:
                    encoder outputs in shape (T*B*H)
                :return
                    decoder output
                Note: we run this one step at a time i.e. you should use a outer loop
                    to process the whole sequence
                Tip(update):
                EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
                different from that of DecoderRNN
                You may have to manually guarantee that they have the same dimension outside this function,
                e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # embedded = 1 * Batch * emdedd_size
        embedded = self.embed(input).unsqueeze(0)
        embedded = self.dropout(embedded)

        attention_weights = self.attention(last_hidden, encoder_outputs)

        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        # 1 * B * emb_size
        context = context.transpose(0, 1)

        # 1 * B * (emb_size+emb_size)
        rnn_input = torch.cat([embedded, context], 2)

        # 1 * B * emb_size
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)

        # 最后需要将attention的结果：重新对encoder的输出进行权重分配计算（1*B*N）【当前解码是如何关注encoder部分的】
        # 当前解码器的输入【attention的结果】得出的hidden结果
        # ----------上述两部分结合起来作为最终预测的输入------------------
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)

        # 当前状态预测的输出、需要传递到下一个状态的隐层值，当前步对于encoder的关注结果
        # output=B*V  hidden = 1 * B *hidden_size attention_weights= B * 1 *T
        return output, hidden, attention_weights
