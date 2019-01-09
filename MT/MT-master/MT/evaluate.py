import random

import torch
from torch.autograd import Variable

from dataset import TextDataset
from model import AttentionDecoderRnn, DecoderRNN, EncoderRnn
import matplotlib.pyplot as plt

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10
use_attention = True
use_cuda = torch.cuda.is_available()
lang_dataset = TextDataset()
print('*'*30)

def evaluate(encoder_model, decoder_model, input_lang, max_length=MAX_LENGTH):
    if use_cuda:
        input_lang = input_lang.cuda()
    input_lang_Variable = Variable(input_lang)
    input_lang_Variable = input_lang_Variable.unsqueeze(0)
    input_length = input_lang_Variable.size(1)
    encoder_hidden = encoder_model.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder_model.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_outputs, encoder_hidden = encoder_model(input_lang_Variable[:, ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoded_attentions = torch.zeros(max_length, max_length)

    if use_attention:
        for di in range(max_length):
            decoder_output, decoder_hidden, decoded_attention = decoder_model(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoded_attentions[di] = decoded_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    else:
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder_model(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    if use_attention:
        return decoded_words, decoded_attentions[:di+1]
    else:
        return decoded_words

def evaluateRandomly(encoder_model, decoder_model, n=10):
    for i in range(n):
        pair_idx = random.choice(list(range(len(lang_dataset))))
        pair = lang_dataset.pairs[pair_idx]
        input_lang, out_lang = lang_dataset[pair_idx]
        print('>', pair[0])
        print('=', pair[1])

        if use_attention:
            output_words, attentions = evaluate(encoder_model, decoder_model, input_lang)
        else:
            output_words = evaluate(encoder_model, decoder_model, input_lang)
        output_sentences = ' '.join(output_words)
        print('<', output_sentences)
        print('')

input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words

encoder = EncoderRnn(input_size, hidden_size)
encoder.load_state_dict(torch.load('./encoder.pth'))

if use_attention:
    decoder = AttentionDecoderRnn(hidden_size, output_size, n_layer=2)
    decoder.load_state_dict(torch.load('./attn_decoder.pth'))
else:
    decoder = DecoderRNN(hidden_size, output_size, n_layers=2)
    decoder.load_state_dict(torch.load('./decoder.pth'))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
evaluateRandomly(encoder, decoder)

if use_attention:
    pair_idx = random.choice(list(range(len(lang_dataset))))
    pairs = lang_dataset[pair_idx]
    print('>')
    print(pairs[0])
    input_lang, output_lang = lang_dataset[pair_idx]
    output_words, attentions = evaluate(encoder, decoder, input_lang)
    # attention matrix show
    plt.matshow(attentions.cpu().numpy())
    plt.show()



