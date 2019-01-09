#!/usr/bin/python
# -*- coding: utf-8 -*-

############################################################
# if batch_size is 1, there must be a dtype error when doing
#   T.grad, this is something about scan func
#   see https://github.com/Theano/Theano/issues/1772
#
# LSTM + cnn
# test1 top-1 precision: 68.3%
############################################################
import matplotlib.pyplot as plt

import myprint
from collections import OrderedDict
import sys, time, random, operator

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano.config.floatX = 'float32'
#TODO change filepath to your local environment
#include train test1 vectors.nobin

sum_lev1={}

def build_vocab():
    # 该模块用来进行对每个训练集　“问题“　中的词进行建立索引
    # {'UNKNOWN': 0, 'be': 1, 'Life': 2, 'Insurance': 3, 'exempt': 4）
    print ('this is build_vocab module......')
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open('train'):
        items = line.strip().split(' ')
        for i in range(2, 3):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    print ('this is build_vocab module......end')
    return vocab

def load_vectors():
    # 将word2Vec训练的词向量总表加载进来，形成字典
    # detriment [-0.281414, 0.326801, 0.489007, -0.004513, -0.314948, 0.501251, 0.115963, 0.439188, -0.125999, 0.102641, 0.290236, 0.236773, 0.22901, -0.149496, 0.190635, 0.229761, 0.252815, -0.515482, -0.205738, 0.173595, -0.594514, 0.20502, 0.880413, -0.407963, 0.462857, -0.640552, 0.389681, 0.252659, 0.678524, 0.034124, -0.313482, -0.18532, -1.153895, 0.412894, 0.185019, -0.234084, 0.051742, -0.10899, 0.121579, 0.175913, 0.494848, -0.250087, -0.074854, -0.299782, -0.220507, -0.309087, -0.051146, 0.04891, 0.011144, -0.231519, 0.20684, -0.025146, 0.377118, 0.573985, -0.208133, -0.378014, 0.388793, -0.454309, -0.056063, 0.217395, -0.057875, -0.065854, -0.291069, 0.295217, 0.25584, -0.204859, 0.073681, -0.184012, 0.108607, -0.097276, -0.394206, -0.091346, 0.454339, -0.177889, -0.143467, -0.109876, -0.523787, 0.32804, -0.771943, -0.13954, 0.250488, 0.379172, -0.026107, 0.210163, -0.358157, -0.426336, 0.01619, 0.053698, 0.149075, -0.508331, -0.035069, -0.305143, -0.042262, 0.158667, -0.506218, -0.128095, -0.16515, 0.344715, 0.160928, 0.093768]
    print ('this is laod_vectors module......')
    vectors = {}
    # for line in open('/export/jw/cnn/insuranceQA/vectors.nobin'):
    for line in open('vectors.nobin'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    print ('this is laod_vectors module......end')
    return vectors

def load_word_embeddings(vocab, dim):
    # 将输入进来的词汇表对应到word2vec词向量表里面，把要训练的词进行词向量表示
    # 如果词表里面没有对应到，那么就是１００维的0.01
    print ('this is load_word_embeddings module.......')
    vectors = load_vectors()
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, dim):
            vec.append(0.01)
        embeddings.append(vec)
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
    print ('this is load_word_embeddings module.......end')
    print("embeddings::::::::::::::")
    print(len(embeddings))
    print(len(embeddings[0]))
    return np.array(embeddings, dtype='float32')

#be attention initialization of UNKNNOW
def encode_sent(vocab, string, size):
    # print ('this is encode_sent module......')
    x, m = [], []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
        if words[i] == '<a>': #TODO
            m.append(1) #fixed sequence length, else use 0
        else:
            m.append(1)
    # x,m都是1*100维（１００维里面的数据对应的是词典的词索引）
    return x, m

def load_train_list():
    # 将训练集中的每一条有四个部分表示成列表存储(列表中的列表存储):[['zhangtaolin', 'in', 'wuhu', '!']]
    print ('this is load_train_list module....')
    trainList = []
    for line in open('train'):
        items = line.strip().split(' ')
        if items[0] == '1':
            trainList.append(line.strip().split(' '))
    print ('this is load_train_list module....end')
    return trainList

def load_test_list():
    print ('this is load_test_list.....')
    testList = []
    for line in open('test1'):
        testList.append(line.strip().split(' '))
    print ('this is load_test_list.....end')
    return testList

def load_data(trainList, vocab, batch_size):
    # 每次运行这个函数，都会在训练数据集中运行２５６次，每次抽取两条数据，随机表示成一个真正抽取的，一个是配合这个真正的，只要那个里面的答案作为计算余弦值的对比
    # 返回的是train_1为256*100维的对真正要抽的数据的问题编码（那100个是对应的词的索引）
    # train_2是真正要抽的数据的答案的编码
    # train_3是配合的那个错误的答案的编码
    # mask是对应的train的里面有多少个<a>，有就补上一个１，没有的话，为了补全100维，设置为０
   # print ('this is load_data module......')
    train_1, train_2, train_3 = [], [], []
    mask_1, mask_2, mask_3 = [], [], []
    counter = 0
    while True:
        # 抽取一个问题：对应了问题和答案　　抽取一个随机的假的问题:把里面的答案作为前面抽取问题的一个假答案
        pos = trainList[random.randint(0, len(trainList)-1)]
        neg = trainList[random.randint(0, len(trainList)-1)]
        if pos[2].startswith('<a>') or pos[3].startswith('<a>') or neg[3].startswith('<a>'):
            # print 'empty string ......'
            continue

        # 返回回来的x为pos里面词对应词表索引数字，如果抽取的词不在里面就默认使用UNKOWN这个词对应的索引数字.
        # m表示的是有多少个<a>,数目就会有多少个1
        # encode_sent编码默认的是问题或者答案都是在100个词以内
        x, m = encode_sent(vocab, pos[2], 100)
        train_1.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, pos[3], 100)
        train_2.append(x)
        mask_2.append(m)
        x, m = encode_sent(vocab, neg[3], 100)
        train_3.append(x)
        mask_3.append(m)
        counter += 1
        if counter >= batch_size:
            break
    #print ('this is load_data module......end')
    # 这里是转置（256*100）-->(100*256)
    return np.transpose(np.array(train_1, dtype=config.floatX)), np.transpose(np.array(train_2, dtype=config.floatX)), np.transpose(np.array(train_3, dtype=config.floatX)), np.transpose(np.array(mask_1, dtype=config.floatX)) , np.transpose(np.array(mask_2, dtype=config.floatX)), np.transpose(np.array(mask_3, dtype=config.floatX))

def load_data_val(testList, vocab, index, batch_size):
    # 这个模块的作用是把对应的测试集形成对应的真正答案的编码（对应的真正答案和问题），还有一个为了衡量答案的效果的冗余随机抽取的答案
    # 如果抽取的测试集的数量不够，就会用最后一个测试集数据进行补全
    # 返回的六个列表与上面函数返回的一样
    #print ('this is load_data_val module......')
    x1, x2, x3, m1, m2, m3 = [], [], [], [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        # testList是把整个测试集（每条测试集是四个部分作为列表中的一个元素）加载成了一个列表　　大小一共是１００００
        if true_index >= len(testList):
            true_index = len(testList) - 1
        items = testList[true_index]
        x, m = encode_sent(vocab, items[2], 100)
        x1.append(x)
        m1.append(m)
        x, m = encode_sent(vocab, items[3], 100)
        x2.append(x)
        m2.append(m)
        x, m = encode_sent(vocab, items[3], 100)
        x3.append(x)
        m3.append(m)
    #print ('this is load_data_val module......end')
    return np.transpose(np.array(x1, dtype=config.floatX)), np.transpose(np.array(x2, dtype=config.floatX)), np.transpose(np.array(x3, dtype=config.floatX)), np.transpose(np.array(m1, dtype=config.floatX)) , np.transpose(np.array(m2, dtype=config.floatX)), np.transpose(np.array(m3, dtype=config.floatX))

def validation(validate_model, testList, vocab, batch_size ,n_ecpoh):
    print ('this is validation module......')
    index, score_list = int(0), []
    while True:
        x1, x2, x3, m1, m2, m3 = load_data_val(testList, vocab, index, batch_size)
        # validate_model输出的是cos12 cos13　所以对应的是batch_scores为每一批数据中问题与正确答案之间的cosine值（２５６个）　　
        # noisee为每一批数据中问题与冗余答案之间的cosine值（２５６个）   这次冗余答案选取的其实是正确答案本身，但是后面没有用到，所以选取什么作为冗余答案无所谓(但是模型定义必须选取)
        batch_scores, noise = validate_model(x1, x2, x3, m1, m2, m3)
        for score in batch_scores:
            score_list.append(score)
        index += batch_size
        if index >= len(testList):
            break
        print ("这是<"+str(n_ecpoh)+">代的测试集： "+' Evaluation ' + str(index))
    sdict, index = {}, int(0)
    for items in testList:
        qid = items[1].split(':')[1]
        if not qid in sdict:
            sdict[qid] = []
        sdict[qid].append((score_list[index], items[0]))
        index += 1
    lev0, lev1 = float(0), float(0)
    of = open(str(n_ecpoh), 'a')
    for qid, cases in sdict.items():
        cases.sort(key=operator.itemgetter(0), reverse=True)
        score, flag = cases[0]
        if flag == '1':
            print("本次问题<"+qid+">找到了正确答案")
            lev1 += 1
        if flag == '0':
            lev0 += 1
    for s in score_list:
        of.write(str(s) + '\n')
    of.write('lev1:' + str(lev1) + '\n')
    of.write('lev0:' + str(lev0) + '\n')
    print ('lev1:' + str(lev1))
    print ('lev0:' + str(lev0))
    of.close()
    sum_lev1[str(n_ecpoh)] = lev1
    print ('this is validation module......end')

def ortho_weight(ndim):
    # 生成对应的ndim方阵的奇异值分解，返回左正交矩阵
    print ('this is ortho_weight module......')
    W = np.random.randn(ndim, ndim)
    # 奇异值分解　U、V是两个正交矩阵　s是中间的sigma对角矩阵
    u, s, v = np.linalg.svd(W)
    print ('this is ortho_weight module......end')
    return u.astype(config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def param_init_cnn(filter_sizes, num_filters, proj_size, tparams, grad_params):
    # 该模块生成的是对应于filter_size范围[1 2 3 5]的权重和偏置  是随机初始化的
    print ('this is param_init_cnn module......')
    # 对于某一个伪随机数发生器，只要该种子（seed）相同，产生的随机数序列就是相同的
    rng = np.random.RandomState(23455)

    # filter_sizes = [1, 2, 3, 5]   num_filters=500  proj_size=100
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
        # prod连乘
        # 100 200 300 500
        fan_in = np.prod(filter_shape[1:])
        # 50000 100000 150000 250000
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        # shared是全局变量，函数共用。
        # uniform生成的事介于【-W_bound,W_bound】之间的一个随机数　形成的矩阵形状如filter_shape
        # 生成的就是filter的权重矩阵
        W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        tparams['cnn_W_' + str(filter_size)] = W
        # 生成的是bias(1*500)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        tparams['cnn_b_' + str(filter_size)] = b
        grad_params += [W, b]
    print ('this is param_init_cnn module......end')
    return tparams, grad_params

def param_init_lstm(proj_size, tparams, grad_params):
    print ('this is param_init_lstm module......')
    # 矩阵的拼接（1表示的是横轴拼接　默认是列拼接）
    # 拼接出来的是100＊400的矩阵  这里应该可以理解成每个句子只有四个词组成
    W = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    W_t = theano.shared(W, borrow=True)
    tparams[_p('lstm', 'W')] = W_t
    # 拼接出来的也是100*400的矩阵
    U = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    U_t = theano.shared(U, borrow=True)
    tparams[_p('lstm', 'U')] = U_t
    # 生成1*400维的矩阵列表
    b = np.zeros((4 * proj_size,))
    b_t = theano.shared(b.astype(config.floatX), borrow=True)
    tparams[_p('lstm', 'b')] = b_t
    grad_params += [W_t, U_t, b_t]
    print ('this is param_init_lstm module......end')
    return tparams, grad_params

def dropout_layer(state_before, use_noise, trng):
    print ('this is dropout_layer module......')
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    print ('this is dropout_layer module......end')
    return proj

class LSTM(object):
  def __init__(self, input1, input2, input3, mask1,mask2, mask3, word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters):
    # sequence_len=100
    # word_embeddings表示的是对训练集的问题词的并集进行词向量表示（列为１００）
    #proj_size means embedding_size
    #'lstm_W' = [embedding_size, embedding_size]
    #'lstm_U' = [embedding_size, embedding_size]
    #'lstm_b' = [embedding_size]
    proj_size = 100 #TODO, what does proj mean
    self.params, tparams = [], {}
    # self.params里面包含　W(100*400) U(100*400) b(1*400) 随机初始化的。
    tparams, self.params = param_init_lstm(proj_size, tparams, self.params)

    # 返回了四种W(500*1*n*100) b(1*500)，因为filter_sizes[1 2 3 5]
    # 100维是一个介于一个固定区间（可调）的随机数
    tparams, self.params = param_init_cnn(filter_sizes, num_filters, proj_size, tparams, self.params)

    lookup_table = theano.shared(word_embeddings, borrow=True)
    tparams['lookup_table'] = lookup_table
    self.params += [lookup_table]
    # ***********************************************************************************************************
    #     params里面存储了lstm的w u b参数　cnn的w b参数(四种)　word_embedding矩阵
    # ***********************************************************************************************************

    n_timesteps = input1.shape[0]
    n_samples = input1.shape[1]

    # 问题
    lstm1, lstm_whole1 = self._lstm_net(tparams, input1, sequence_len, batch_size, embedding_size, mask1, proj_size)
    # 正确答案
    lstm2, lstm_whole2 = self._lstm_net(tparams, input2, sequence_len, batch_size, embedding_size, mask2, proj_size)
    # 随机抽取的冗余对比答案
    lstm3, lstm_whole3 = self._lstm_net(tparams, input3, sequence_len, batch_size, embedding_size, mask3, proj_size)

    #dimshuffle [sequence_len, batch_size, proj_size] to [batch_size, sequence_len, proj_size]
    cnn_input1 = T.reshape(lstm1.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input2 = T.reshape(lstm2.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input3 = T.reshape(lstm3.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])

    # cnn1 cnn2 cnn3都是对每个问题的词向量表示
    cnn1 = self._cnn_net(tparams, cnn_input1, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn2 = self._cnn_net(tparams, cnn_input2, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn3 = self._cnn_net(tparams, cnn_input3, batch_size, sequence_len, num_filters, filter_sizes, proj_size)

    len1 = T.sqrt(T.sum(cnn1 * cnn1, axis=1))
    len2 = T.sqrt(T.sum(cnn2 * cnn2, axis=1))
    len3 = T.sqrt(T.sum(cnn3 * cnn3, axis=1))

    self.cos12 = T.sum(cnn1 * cnn2, axis=1) / (len1 * len2)
    self.cos13 = T.sum(cnn1 * cnn3, axis=1) / (len1 * len3)

    zero = theano.shared(np.zeros(batch_size, dtype=config.floatX), borrow=True)
    # margin = 1*256
    margin = theano.shared(np.full(batch_size, 0.05, dtype=config.floatX), borrow=True)
    # diff 1*256
    diff = T.cast(T.maximum(zero, margin - self.cos12 + self.cos13), dtype=config.floatX)
    # cost是一批数据集（２５６）的损失结果综合
    self.cost = T.sum(diff, acc_dtype=config.floatX)
    # accuracy是一批（２５６）的准确率
    self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)

  def _cnn_net(self, tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, proj_size):
    print ('this is _cnn_net module......')
    outputs = []
    # 1 2 3 5
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
        image_shape = (batch_size, 1, sequence_len, proj_size)
        W = tparams['cnn_W_' + str(filter_size)]
        b = tparams['cnn_b_' + str(filter_size)]
        conv_out = conv2d(input=cnn_input, filters=W, filter_shape=filter_shape, input_shape=image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
        outputs.append(pooled_active)
    num_filters_total = num_filters * len(filter_sizes)
    output_tensor = T.reshape(T.concatenate(outputs, axis=1), [batch_size, num_filters_total])
    print ('this is _cnn_net module......end')
    return output_tensor

  def _lstm_net(self, tparams, _input, sequence_len, batch_size, embedding_size, mask, proj_size):
    # sequence_len=100
    # prij_size = 100
    # input = 100 * 256(词的索引)
    # mask = 100*256(1或者０)
    print ('this is _lstm_net module......')
    input_matrix = tparams['lookup_table'][T.cast(_input.flatten(), dtype="int32")]

    # 调整输入尺寸大小变成:100*(256*100)
    input_x = input_matrix.reshape((sequence_len, batch_size, embedding_size))
    proj, proj_whole = lstm_layer(tparams, input_x, proj_size, prefix='lstm', mask=mask)
    #if useMask == True:
    #proj = (proj * mask[:, :, None]).sum(axis=0)
    #proj = proj / mask.sum(axis=0)[:, None]
    #if options['use_dropout']:
    #proj = dropout_layer(proj, use_noise, trng)
    print ('this is _lstm_net module......end')
    return proj, proj_whole

#state_below is word_embbeding tensor(3dim)
def lstm_layer(tparams, state_below, proj_size, prefix='lstm', mask=None):
    # state_below 100*256*100
    print ('this is lstm_layer module......')
    #dim-0 steps, dim-1 samples(batch_size), dim-3 word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #h means hidden output? c means context（细胞状态）? so we'll use h?
    #rval[0] = [sequence_len, batch_size, proj_size], rval[1] the same

    #so preact size must equl to x_(lstm input slice)
    #if you want change lstm h(t) size, 'lstm_U' and 'lstm_b'
    #and precat must be changed to another function, like h*U+b
    #see http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #f(t) = sigmoid(Wf * [h(t-1),x(t)] + bf)
    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, proj_size))
        f = T.nnet.sigmoid(_slice(preact, 1, proj_size))
        o = T.nnet.sigmoid(_slice(preact, 2, proj_size))
        c = T.tanh(_slice(preact, 3, proj_size))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        #if mask(t-1)==0, than make h(t) = h(t-1)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    # w 100*400   U 100*400  b 1*400
    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = proj_size
    # 迭代函数：可以保存之前的所有结果。
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    print ('this is lstm_layer module......end')
    return rval[0], rval[1]

def _p(pp, name):
    return '%s_%s' % (pp, name)

def train():
    print ('this is train......')
    batch_size = int(500)
    embedding_size = 100
    learning_rate = 0.05
    n_epochs = [5,10,20,40,50]
    validation_freq = [5,10,20,40,50]
    filter_sizes = [1, 2, 3, 5]
    num_filters = 500

    # 对训练集中问题部分的词进行了词索引的建立
    vocab = build_vocab()

    # 对训练集的问题词（取并集）进行了word2vec向量表示
    word_embeddings = load_word_embeddings(vocab, embedding_size)

    # trainList=18540*4
    trainList = load_train_list()
    # 10000*4
    testList = load_test_list()

    # 这里提取了２５６个训练数据集的编码,编码的数字是词索引
    # 六个都是100*256的矩阵（１００是对应的词索引，２５６对应的是随机抽取的问题和一个冗余答案）
    train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_data(trainList, vocab, batch_size)
    x1, x2, x3 = T.fmatrix('x1'), T.fmatrix('x2'), T.fmatrix('x3')
    m1, m2, m3 = T.fmatrix('m1'), T.fmatrix('m2'), T.fmatrix('m3')
    model = LSTM(
        input1=x1, input2=x2, input3=x3,
        mask1=m1, mask2=m2, mask3=m3,
        word_embeddings=word_embeddings,
        batch_size=batch_size,
        sequence_len=train_x1.shape[0], #行数100
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters)

    # cost是一批数据集（２５６个）的损失总和
    # cost12是计算一批数据集（２５６）中的每一个问题和每一个问题的正确答案之间的cosine值
    # cost1３是计算一批数据集（２５６）中的每一个问题和每一个问题的随机抽取的冗余答案之间的cosine值
    cost, cos12, cos13 = model.cost, model.cos12, model.cos13
    # accuracy是一批数据集的准确率总和的计算结果
    params, accuracy = model.params, model.accuracy
    grads = T.grad(cost, params)
    updates = [
        # SGD方法
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    p1, p2, p3 = T.fmatrix('p1'), T.fmatrix('p2'), T.fmatrix('p3')
    q1, q2, q3 = T.fmatrix('q1'), T.fmatrix('q2'), T.fmatrix('q3')



    #  function是一个由inputs计算outputs的对象，它关于怎么计算的定义一般在outputs里面，这里outputs一般会是一个符号表达式。
    # updates: 这里的updates存放的是一组可迭代更新的量
    # givens这里存放的也是一个可迭代量，可以是列表，元组或者字典，即每次调用function，givens的量都会迭代变化   givens可以用inputs参数来代替，但是givens的效率更高。
    train_model = theano.function(
        [p1, p2, p3, q1, q2, q3],
        [cost, accuracy],
        updates=updates,
        givens={
            x1: p1, x2: p2, x3: p3, m1: q1, m2: q2, m3: q3
        }
    )

    v1, v2, v3 = T.matrix('v1'), T.matrix('v2'), T.matrix('v3')
    u1, u2, u3 = T.matrix('u1'), T.matrix('u2'), T.matrix('u3')
    validate_model = theano.function(
        inputs=[v1, v2, v3, u1, u2, u3],
        outputs=[cos12, cos13],
        #updates=updates,
        givens={
            x1: v1, x2: v2, x3: v3, m1: u1, m2: u2, m3: u3
        }
    )

    # epoch = 0
    # done_looping = False
    for i in range(len(n_epochs)):
        epoch = 0
        done_looping = False
        while (epoch < n_epochs[i]) and (not done_looping):
            epoch += 1

            # 每一代都是处理256个随机问题，都需要重新进行随机抽取100*256
            train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_data(trainList, vocab, batch_size)
            cost_ij, acc = train_model(train_x1, train_x2, train_x3, mask1, mask2, mask3)
            print ('load data done ...... epoch:' + str(epoch) + ' cost损失总和:' + str(cost_ij) + ', acc准确率' + str(acc))
            if epoch % validation_freq[i] == 0:
                myprint.my_print()
                print ('运行测试集 ......')
                validation(validate_model, testList, vocab, batch_size,n_epochs[i])
                print('测试集运行完毕 ......')
                myprint.my_print()
        print ('this is train......end')
if __name__ == '__main__':
    train()
    myprint.my_print()
    print("最终结果为:")
    print(sum_lev1)
    myprint.my_print()
    print('开始绘图..')
    X = ['5','10','20','40','50']
    Y = []
    Y.append(sum_lev1.get('5'))
    Y.append(sum_lev1.get('10'))
    Y.append(sum_lev1.get('20'))
    Y.append(sum_lev1.get('40'))
    Y.append(sum_lev1.get('50'))
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("bar chart")

    plt.show()
    plt.savefig("barChart.jpg")
