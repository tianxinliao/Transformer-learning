import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
import matplotlib.pyplot as plt
import copy
# import seaborn



# input embedding
class Embedding(nn.Module):
    def __init__(self, d_model, voacb):
        # d_model=512，模型维度
        # vocab=源语言词表大小
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(voacb, d_model)
        # one-hot转embedding
        # 此时得到的是词语的embedding矩阵 (vocab, d_model)
        # 这里是随机初始化的
        # 所以这是个待训练的矩阵
        self.d_model = d_model

    def forward(self, x):
        # x (batch_size, sequence_length)
        # one-hot 大小等于当前词表大小
        return self.lut(x) * math.sqrt(self.d_model)
        # 得到(vocab, 512)的embedding矩阵
        # 乘以一个512的开方，不知道干嘛用的
        # 这里的输出格式为 (batch_size, sequence_length, 512)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model = 512
        # dropout = 0.1
        # max_len 代表事先准备好长度为5000的序列的位置编码
        # 其实没必要，一般一两百就够了
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # --- 这部分计算位置编码，公式参考论文 ---
        pe = torch.zeros(max_len, d_model)
        # (5000, 512) 矩阵，保存每个位置的位置编码，一共5000个位置
        # 每个位置用一个和词的embedding维度相同的向量（512）来表示
        # 方便后面相加
        position = torch.arange(0, max_len).unsqueeze(1)
        # (5000) -> (5000, 1)
        div_term = torch.ones(int(d_model / 2)) / (10000.0 ** (torch.arange(0, d_model, 2) / float(d_model)))
        # 位置编码sin、cos临边固定的那个部分
        # 一共(0, 2, ..., 510)，256个值，分别对应512维的奇数维和偶数维
        # The annotated Transformer 里边用的是下边这个很魔幻的写法
        # div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 这个写法对公式进行了数学变形
        # 不知道好处是什么
        pe[:, 0::2] = torch.sin(position * div_term)
        # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)
        # 奇数维度
        pe = pe.unsqueeze(0)
        # (5000, 512) -> (1, 5000, 512) 为batch_size留出位置
        self.register_buffer('pe', pe)
        # 位置编码是不会更新的，是写死的，所以这个class里没有可训练的参数
        # --- 这部分计算位置编码，公式参考论文 ---

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        # 这一层接收的x是input_embedding层的输出结果
        # 然后把自己的位置编码，进行对位相加
        # x (30, 10, 512)
        # self.pe[:, :x.size(1)] 就取的是min(10, 5000)
        # 在一个batch中的30个序列，加的都是一样的值
        return self.dropout(x)
        # 增加一个dropout操作


# attention(Q,K,V) = softmax(QK^T/sqrt(d_k))*V
def attention(query, key, value, mask=None, dropout=None):
    # 在每个注意力头上
    # query, key, value 的形状类似于(30, 8, 10, 64),(30, 8, 11, 64),(30, 8, 11, 64)
    # 30 = batch_size
    # 8 = head_num
    # 10 = 目标序列中词的个数
    # 11 = 源序列中词的个数
    # （这里预设的场景是Decoder端，query来自目标端，key和value来自源端）
    d_k = query.size(-1)
    # q,k,v矩阵维度
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    # (30, 8, 10, 64)和(30, 8, 11, 64)的转置相乘
    # (30,8,10,11)
    # 代表10个目标语言序列中的每个词和11个源语言序列中的词分别的“相关度”
    # 除以sqrt(d_k)，防止出现过大的相关度，让梯度更稳定
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 使用mask，对已经计算好的scores，按照mask矩阵，填上-1e9
        # 在下一步计算softmax的时候，被设置成-1e9的数对应值约等于0，被忽视
    p_attn = F.softmax(scores, dim=-1)
    # 对score的最后一维做softmax
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 神经网络的深度copy
# module是要copy的模型，N是copy的份数
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h=8, d_model=512
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 注意力头必须是维度的倍数（词向量才能被正好分成n个头）
        self.d_k = d_model // h
        # d_k = 512//8 = 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 定义四个线性层，每个大小是(512, 512)
        # 每个线性层里边有两类训练参数
        # weight (512, 512)
        # bias (512)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query (30, 10, 512) 源端seq长度10
        # key (30, 11, 512) 目标端端seq长度11
        # value (30, 11, 512) 目标端端seq长度11

        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0) # 30

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # zip 把linears和(q, k, v)中相应元素打包成元组
        # l, x 分别去了linears中的前三个线性层和q、k、v
        # 针对 q, k, v 三个矩阵的操作
        # query (30, 10, 512) -> (30, 10, 8, 64) -> (30, 8, 10, 64)
        # key (30, 11, 512) -> (30, 8, 11, 64)
        # value (30, 11, 512) -> (30, 8, 11, 64)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 调用上面的attention函数
        # 输出形状和输入形状相同
        # attn的形状为(30, 8, 10, 11)
        # 意义：对于一个batch中的30个样本，在8个头中，对于目标端的10个词，源端的11个词分别的注意力

        x = x.transpos(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #  (30, 8, 10, 64) -> (30, 10, 512)

        return self.linears[-1](x)
        # 执行最后一个线性层，把(30, 10, 512)过一次线性层
        # (30, 10, 512)


# LN对每个sublayer的输出进行处理，也对6层EncoderLayer之后的输出进行处理
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features=512=d_model=512
        # eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # a_2是一个可训练参数向量,(512)
        self.b_2 = nn.Parameter(torch.zeros(features))
        # b_2也是一个可训练参数向量,(512)
        self.eps = eps

    def forward(self, x):
        # x,(batch_size, sequence_length, 512)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度取平均值
        # (batch_size, sequence_length)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差
        # (batch_size, sequence_length)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上是标准化 (x-mean)/std
        # 这里加入了两个可训练向量
        # 分母上加了一个epsilon，用来防止分母为0


# 残差Add + Norm
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        # size=d_model=512
        # droupout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) # 512，用于定义a_2和b_2
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # x,(batch_size, sequence_length, 512)
        # sublayer 是一个具体的 MultiHeadAttention 或者 PositionwiseFeedForward 对象
        return x + self.dropout(sublayer(self.norm(x)))
        # x (30, 10, 512) -> layernorm -> (30, 10, 512)
        # -> sublayer(MultiHeadAttention or PositionwiseFeedForward)
        # -> (30, 10 512) -> dropout -> (30 ,10 512)
        # 然后加上输入的x（实现残差相加）
# 这个类本身没有自己的可训练参数，self.norm中有1024个


# 这就是个全连接层
# 其实现的是
# FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
# W_1 和 b_1 是第一个线性层的参数
# max(0, xW_1 + b_1) 是 relu 激活函数
# W_2 和 b_2 是第二个线性层的参数
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512 * 4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层 (512, 2048)
        # 其中有两种可训练参数
        # weights 矩阵 (512, 2048)
        # biases 偏置 (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层 (2048, 512)
        # 其中有两种可训练参数
        # weights 矩阵 (2048, 512)
        # biases 偏置 (512)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x (batch_size, sequence_len, 512)
        # (30, 10, 512)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)
        # -> relu -> (30, 10, 2048)
        # -> dropout -> (30, 10, 2048)
        # -> self.w_2 -> (30, 10, 512)


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        # size = d_model = 512
        # self_attn = MultiHeadAttention对象，first sublayer
        # feed_forward = PositionalwiseFeedForward对象，second sublayer
        # dropout = 0.1
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # x (30, 10, 512)
        # mask 是 (batch_size, 10, 10) 的矩阵
        # 类似于对于当前这个词w_i，有其他哪些词w_j是可见的
        # 做self-attention的时候，所有其他词都是可见的，除了"<blank>"这样的填充词
        # 做source-target attention的时候，除了w左边的词，都是可见的
        # （后面的词相当于参考答案，提前看了参考答案会导致模型训练不当）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # x (30, 10, 512) -> self_attn (MultiHeadAttention) -> (30, 10, 512)
        # -> sublayerConnection -> (30, 10 512)
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    # Encoder是N个的EncoderLayer层的叠加
    def __init__(self, layer, N):
        # layer = EncoderLayer, N = 6
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # x (30, 10, 512)
        # (batch_size, sequence_len, d_model)
        # mask 是类似于 (batch_size, 10, 10) 的矩阵
        # 为什么是 10 * 10
        # encoder是 self-attention
        # 一个batch里边不一定所有句子都是等长的，比如有的没到10，要用<blank>补齐
        # 这时候mask矩阵就是把用<blank>补齐的部分遮掉
        # 在计算loss的时候不会算进去
        for layer in self.layers:
            x = layer(x, mask)
            # 把x通过6层的encoderLayer
        return self.norm(x)
        # 6层encoderLayer出来的结果还要最后做一次LayerNorm
        # (30, 10, 512)


class DecoderLayer(nn.Module):
    # 从Transformer的模型图可以看出
    # DecoderLayer是由self-attention层 + src-trg-attention层 + 全连接层组成的
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size = d_model = 512
        # self_attn = MultiHeadAttention 目标语言序列的自注意力机制
        # src_attn = MultiHeadAttention 目标语言和源语言之间的注意力
        # query来自目标语言，key和value来自源语言
        # feed_forward 全连接层
        # dropout = 0.1
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # (batch_size, sequence_len, 512)
        # 来自源语言序列的Encoder之后的输出，作为memory供目标语言序列检索匹配
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过匿名函数实现目标序列的自注意力编码
        # 然后把结果放入残差+Norm网络
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 通过第二个匿名函数，实现目标序列与原序列的注意力计算
        # 结果放入残差+Norm网络
        return self.sublayer[2](x, self.feed_forward)
        # 过一个全连接层
        # 再过一个残差+Norm网络


class Decoder(nn.Module):

    def __init__(self, layer, N):
        # layer = DecoderLayer, N = 6
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # 过6层的DecoderLayer
        return self.norm(x)


class Generator(nn.Module):
    # 最后生成输出序列的层
    # 由一个线性层 + 一个softmax层组成
    def __init__(self, d_model, vocab):
        # d_model = 512
        # vocab = 目标语言词表大小
        super(Generator, self).__init__()
        self.proj = nn.LayerNorm(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        # x (batch_size, sequence_len, 512) -> 全连接层 -> (30, 10, trg_vocab_size)
        # 对最后一个维度计算log，然后softmax
        # 得到(30, 10, trg_vocab_size)


# 最外边的EncoderDecoder类
# 由上面写的模块组成
class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        # 源端编码，word embedding + positional embedding
        self.tgt_embed = tgt_embed
        # 目标端编码，word embedding + positional embedding
        self.generator = generator
        # 生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decod

    def encode(self, src, src_mask):
        # src = (batch_size, sequence_len)
        # 这个src就是输入端的句子的one-hot表示
        return self.encoder(self, self.src_embed(src), src_mask)
        # 对源端序列进行编码
        # 得到(batch_size, sequence_len, 512)的tensor

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self, self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # 对目标序列进行编码
        # 得到(batch_size, sequence_len, 512)的tensor

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    # 先对源语言序列进行编码，把结果作为memory传给目标语言的编码器


def subsequent_mask(size):
    # e.g. size = 10
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # [[[0 1 1 1 1 1 1 1 1 1]
    #   [0 0 1 1 1 1 1 1 1 1]
    #   [0 0 0 1 1 1 1 1 1 1]
    #   [0 0 0 0 1 1 1 1 1 1]
    #   [0 0 0 0 0 1 1 1 1 1]
    #   [0 0 0 0 0 0 1 1 1 1]
    #   [0 0 0 0 0 0 0 1 1 1]
    #   [0 0 0 0 0 0 0 0 1 1]
    #   [0 0 0 0 0 0 0 0 0 1]
    #   [0 0 0 0 0 0 0 0 0 0]]]
    return torch.from_numpy(subsequent_mask) == 0
    # [[[ True False False False False False False False False False]
    #   [ True  True False False False False False False False False]
    #   [ True  True  True False False False False False False False]
    #   [ True  True  True  True False False False False False False]
    #   [ True  True  True  True  True False False False False False]
    #   [ True  True  True  True  True  True False False False False]
    #   [ True  True  True  True  True  True  True False False False]
    #   [ True  True  True  True  True  True  True  True False False]
    #   [ True  True  True  True  True  True  True  True  True False]
    #   [ True  True  True  True  True  True  True  True  True  True]]]


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model) # 8, 512
    # 构造一个多头注意力层

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 512, 2048, 0.1
    # 构造一个 feed forward 对象

    position = PositionalEncoding(d_model, dropout)
    # 位置编码

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # 对参数进行随机初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model






if __name__ == "__main__":
    tmp_model = make_model(30000, 30000, 6)
    # src_vocab = 3w
    # trg_vocab = 3w
    # num of EncoderLayer and DecoderLayer = 6

    for name, param in tmp_model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
        else:
            print('no gradient necessary', name, param.data.shape)
