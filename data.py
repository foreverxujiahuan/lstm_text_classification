'''
@Autor: xujiahuan
@Date: 2020-04-29 23:11:58
@LastEditors: xujiahuan
@LastEditTime: 2020-04-30 21:52:20
'''
import collections
import os
import random
import tarfile
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm


# 将压缩文件aclImdb_v1.tar.gz进行解压，解压在同级目录下
DATA_ROOT = "data"
fname = 'data/aclImdb_v1.tar.gz'
# 判断是否解压完成，如果解压完成，就不进行解压了
if not os.path.exists(os.path.join(os.path.join(DATA_ROOT, "aclImdb"))):
    print("从压缩包解压...")
    with tarfile.open(fname, 'r') as f:
        f.extractall(DATA_ROOT)


# 获取训练集和测试集
def read_imdb(folder='train', data_root="/S1/CSCL/tangss/Datasets/aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        # 遍历文件夹下所有文件,tqdm为显示遍历进度的包
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                # 将句子中的回车去除
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    # 最终返回的data是一个List，里面存放了所有folder下的数据，每个数据还有这个句子的所有单词和标签
    return data


data_root = os.path.join(DATA_ROOT, "aclImdb")
train_data, test_data = read_imdb('train', data_root), read_imdb('test',
                                                                 data_root)


# 数据预处理
def get_tokenized_imdb(data):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    # 获得按照空格分开后的所有单词
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    # counter是这个数据里所有单词的出现次数
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 返回一个vocab，获取counter里单词数量大于等于5的数据
    return Vocab.Vocab(counter, min_freq=5)


vocab = get_vocab_imdb(train_data)


def preprocess_imdb(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    # tokenized_data是数据按照空格分开后的句子，是一个二维list
    tokenized_data = get_tokenized_imdb(data)
    # features是每个词在字典中的value
    features = torch.tensor([pad([vocab.stoi[word] for word in words])
                             for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


# 创建数据迭代器
batch_size = 64
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
# 其中每个数据集都有64个句子
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)


def main():
    # 查看数据类型
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        print(X)
        print('-'*100)
        print(y)
        break


if __name__ == "__main__":
    main()
