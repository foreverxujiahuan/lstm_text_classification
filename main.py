'''
@Autor: xujiahuan
@Date: 2020-04-29 23:12:20
@LastEditors: xujiahuan
@LastEditTime: 2020-04-30 22:46:52
'''
import torch
from torch import nn as nn
from model import BiRNN
from data import vocab
import torchtext.vocab as Vocab
import os
from data import DATA_ROOT
from data import train_iter, test_iter
import time


# 判定是否能用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置词嵌入维度、隐藏层神经元数量、隐藏层数量
embed_size, num_hiddens, num_layers = 300, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
# 加载维基百科预训练词向量(使用fasttext),cache为保存目录
fasttext_vocab = Vocab.FastText(cache=os.path.join(DATA_ROOT, "fasttext"))


def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    # 初始化为0
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed


# fastext_vocab是预训练好的词向量
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos,
                                                          fasttext_vocab))
# 直接加载预训练好的, 所以不需要更新它
net.embedding.weight.requires_grad = False


# 训练并且评价模型
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) ==
                            y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑G
                # 如果有is_training这个参数PU
                if('is_training' in net.__code__.co_varnames):
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).
                                argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_ = loss(y_hat, y)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            train_l_sum += loss_.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,\
                 time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                 test_acc, time.time() - start))


lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                             net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence],
                            device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'


def main():
    predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
    predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])


if __name__ == "__main__":
    main()
