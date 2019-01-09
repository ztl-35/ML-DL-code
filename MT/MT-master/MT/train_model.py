import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import DecoderRNN, EncoderRnn, AttentionDecoderRnn

torch.cuda.set_device(1)

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10
lang_dataset = TextDataset()
# 每次枚举都是一个原文，一个翻译
lang_dataloder = DataLoader(lang_dataset, shuffle=True)

# 4489
input_size = lang_dataset.input_lang_words
hidden_size = 256
# 2923
output_size = lang_dataset.output_lang_words
total_epoch = 20

encoder_model = EncoderRnn(input_size, hidden_size)
decoder_model = DecoderRNN(hidden_size, output_size, n_layers=2)
attention_decoder = AttentionDecoderRnn(hidden_size, output_size, n_layers=2)
use_attention = True

if torch.cuda.is_available():
    encoder_model = encoder_model.cuda()
    decoder_model = decoder_model.cuda()
    attention_decoder =attention_decoder.cuda()

def showPlot(points):
    plt.figure()
    x = np.arange(len(points))
    plt.plot(x, points)
    plt.show()

def train(encoder, decoder, total_epoch, use_attention):

    param = list(encoder.parameters()) + list(decoder.parameters())
    # param 一起进行训练
    optimizer = optim.Adam(param, lr=0.001)
    loss_function = nn.NLLLoss()
    plot_loss = []
    for epoch in range(total_epoch):
        start_time = time.time()
        running_loss = 0
        print_loss_total = 0
        total_loss = 0
        for i, data in enumerate(lang_dataloder):
            # input_lang, output_lang = 分别取值（6个 7个等等），不一定，但是都是一句代翻译的话，对应的索引
            input_lang, output_lang = data
            if torch.cuda.is_available():
                # batch=1, length
                input_lang = Variable(torch.tensor(input_lang)).cuda()
                output_lang = Variable(torch.tensor(output_lang)).cuda()

            # 用来保存每次单元输出的情况 n_sqe(Max_length) * hidden_size
            # 10 * 256
            encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))

            if torch.cuda.is_available():
                encoder_outputs = encoder_outputs.cuda()

            # 1*1*256
            encoder_hidden = encoder.initHidden()
            for ei in range(input_lang.size(1)):
                # 挨个输入对应一句代翻译的话的索引 ： input = x(索引), encoder = 256（向量）
                # 1*1*256 1*1*256
                encoder_output, encoder_hidden = encoder(input_lang[:, ei], encoder_hidden)
                # encoder_output 是每一次的输出，最终应该是 n_sqe * hidden_size
                encoder_outputs[ei] = encoder_output[0][0]
            decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()

            # *********************************************************************************
            #     编码的输出保存在encoder_outputs里面{10 * 256 不够最大长度的用0补齐}  最后上下文隐层向量保存在encoder_hidden 1 * 1* 256
            # *********************************************************************************

            decoder_hidden = encoder_hidden

            loss = 0

            if use_attention:
                for di in range(output_lang.size(1)):
                    # 1 * 2925(预测的词表概率值)   1 * 1 * 256（准备传入到下一个词的隐藏值）  1 * 10（本次的权值结果）
                    decoder_output, decoder_hidden, decoder_attention = attention_decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # NLLOSS 用于多分类的对数似然函数
                    loss += loss_function(decoder_output, output_lang[:, di])

                    # 输出当前的最大可能值对应的索引，用于传入到下一步迭代
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    if torch.cuda.is_available():
                        decoder_input = decoder_input.cuda()

                    # 有可能提前终止(多对多的类型是不一定长度)，但最多不超过翻译文本的最大长度
                    if ni == EOS_TOKEN:
                        break
            else:
                for di in range(output_lang.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += loss_function(decoder_output, output_lang[:, di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    if torch.cuda.is_available():
                        decoder_input = decoder_input.cuda()
                    if ni == EOS_TOKEN:
                        break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            print_loss_total += loss.data
            total_loss += loss.data
            if (i + 1) % 5000 == 0:
                print('{}/{}, Loss:{:.6f}'.format(
                    i + 1, len(lang_dataloder), running_loss / 5000))
                running_loss = 0
            if (i + 1) % 100 == 0:
                plot_losses = print_loss_total / 100
                plot_loss.append(plot_losses)
                print_loss_total = 0
        end_time = time.time()
        print('Finish {}/{} , Loss:{:.6f}, Time:{:.0f}s'.format(
            epoch + 1, total_epoch, total_loss / len(lang_dataset), end_time-start_time))

    showPlot(plot_loss)

if use_attention:
    train(encoder_model, attention_decoder, total_epoch, use_attention=True)
else:
    train(encoder_model, decoder_model, total_epoch, use_attention=True)
print('finish training!')
if use_attention:
    torch.save(encoder_model.state_dict(), './encoder.pth')
    torch.save(attention_decoder.state_dict(), './attn_decoder.pth')
else:
    torch.save(encoder_model.state_dict(), './encoder.pth')
    torch.save(decoder_model.state_dict(), './decoder.pth')