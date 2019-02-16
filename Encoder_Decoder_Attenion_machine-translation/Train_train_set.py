import torch
import math
from torch.nn import functional as F

# '这个模块主要用来进行一次epoch的循环训练'

def train(model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    # 设置模型处于训练状态
    model.train()

    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda("0"), trg.cuda("0")

        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        # 更新梯度(梯度爆炸后，就需要进行裁剪)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm(model.paramters(), grad_clip)

        optimizer.step()

        total_loss += loss.data[0]

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0

