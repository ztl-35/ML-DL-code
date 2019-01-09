import re
import unicodedata

import torch
from torch.utils.data import Dataset

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

# 统计输入或需要翻译的语言的一些个数信息
class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始值为前面的两个
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 将data目录下 某语言到某语言的文件加载到内存中来
def readLangs(lang1, lang2, reverse=False):
    # lang1 = English lang2 = fra
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 将每一行的训练文本数据进行转化  形成每行两列的数据，分别表示输入-》输出语言的处理，去掉标点符号，同时转化为小写
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 将哪个语言作为开始翻译的语言，同时将pairs中保留的翻译语言对加载进来
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        # 初始化一个语言对象（当前初始化只有名字，没有统计信息）
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # 返回 English fra对象
    return input_lang, output_lang, pairs

# 英语前缀
eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):
    # 判断这句话处理的条件时：这句话必须小于十个字，同时输出语言（下面函数可以看出是英语）英语必须是包含以下开头的
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    # input_lang = English output_lang = fra
    print('*'*30)
    print('进行词汇数据集的训练准备。。。')
    # reverse=True：lang1是输出语言/lang2时输入语言
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    # 这一步开始把原本只有语言名字的对象 进行文字信息填写
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print('input lang word count: ', len(input_lang.word2count))
    print('output lang word count: ', len(output_lang.word2count))

    # 输入语言的一个class对象，和输出语言的class对象
    # 过滤之后的pairs对
    print('词汇数据集准备完毕...')
    print('*'*30)
    return input_lang, output_lang, pairs

# 返回一个句子的索引表示
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# 将一个句子写成索引表示，同时转化为tensor的变量
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes)
    return result

# 将输入和输出的翻译的训练的句子同时转化为tensor变量
def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    # 每个翻译的是 训练句子个数×长度（索引表示）
    return input_tensor, target_tensor


class TextDataset(Dataset):
    def __init__(self, dataload=prepareData, lang=['eng', 'fra']):
        # 这里就把fra当成输入语言，英语当成输出翻译语言
        self.input_lang, self.output_lang, self.pairs = dataload(
            lang[0], lang[1], reverse=True)
        self.input_lang_words = self.input_lang.n_words
        self.output_lang_words = self.output_lang.n_words

    def __getitem__(self, index):
        # 返回第index个翻译的句子对的tensor表示
        return tensorFromPair(self.input_lang, self.output_lang,
                              self.pairs[index])

    def __len__(self):
        return len(self.pairs)