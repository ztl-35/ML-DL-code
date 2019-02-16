import torch.optim as optim
from Train_train_set import *
from Evaluate_val_set import *
from Hyperparams import *
from Sqe2Sqe import *
from Encoder import *
from Decoder import *
from Utils import *
import os
import math


def main():
    # 在这里设置想要的超参数
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)

    encoder = Encoder(de_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size, n_layers=1, dropout=0.5)
    seq2seq = Sqe2Sqe(encoder, decoder).cuda()

    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)