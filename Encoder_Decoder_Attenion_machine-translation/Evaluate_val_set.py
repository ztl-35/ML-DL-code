from torch.nn import functional as F
from torch.autograd import Variable

def evaluate(model, val_iter, vocub_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocub_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        total_loss += loss.data[0]

    # 用于验证集和测试集上的结果输出
    return total_loss / len(val_iter), output
