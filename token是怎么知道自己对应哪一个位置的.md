---
title: 面试记录：token是怎么知道自己对应哪一个位置的
date: 2025-08-26 16:51:30
categories:
  - interview
tags:
  - LLM
  - interview
mathjax: true
---

# 前言

在[上一篇文章](/2025/08/26/interview/怎么理解多头注意力机制/)中，因为多头注意力机制，下一个词的生成可以并行计算，为显卡赋能带来了方便。但是，并行的时候，如何才能让`token`知道它本来应该在哪？

<!-- more -->

# Transformer

让我们回到`Transformer`的那张超级经典的图中。

![Transformer](http://images.sakebow.cn/interview/transformer.png)

不难发现，在`Encoder`开始之前，就在词向量中嵌入了一层位置向量。

# 位置向量

最开始的时候，文本经过`BPE`分词器，可以得到`input_ids`。具体而言呢，在这里主要是使用`AutoTokenizer`的实例化对象，配合`add_special_tokens`参数，对文本分词，并确认是否在其中加入一些特殊编码，包括暂停、结束等。

然后，有一个词表嵌入矩阵。这个词表嵌入矩阵本质上就是一个可学习的参数矩阵，可以简单地理解为我们读取的大模型参数。一开始矩阵是随机初始化的，在逐步推进的训练中慢慢优化，逐渐贴合数据集。需要注意的是，由于网络结构的不同，我们在使用现有词表嵌入矩阵的时候，必须明确任务类别，并加以区分。如果你在使用`BERT`，那么这个矩阵就是`BERT`的参数矩阵；如果你在使用`GPT`，那么这个矩阵就是`GPT`的参数矩阵。

使用这个词表嵌入矩阵，将`input_ids`中的每个词映射为向量`input_emb`。这个使用的是`PyTorch`的`nn.Embedding`。在这个`input_emb`基础上，需要增加一些位置向量。

当然，位置向量也是有流派的。

## APE

绝对位置嵌入（`absolute position embedding`）首先建立一个可训练表`P`，第`t`个位置直接查表`P[t]`，然后加到`input_emb`中。这个`input_emb`中携带的`P`的训练参数，在后续的迭代过程中逐步更新。

如果不训练，那就是`transfomer`原版，采用一个固定的表`P`，偶数位置用正弦，奇数位置用余弦。

## RPE

相对位置嵌入（`relative position embedding`）与`APE`的不同之处在于，取缔了查询第`t`个位置的信息，而是直接采用`i`与`j`之间的距离，让模型关注更近的对象。

如果是采用相对位置偏置，也就是直接在注意力分数里面加一个与距离相关的标量，类似`ALiBi`的线性斜率，大概是这样：

$$\text{score}\leftarrow\text{score}+\mathbf{b}_h(i-j)\tag{1}$$

$\mathbf{b}_h$指的就是多头注意力机制中每个头的斜率，是一个矩阵。

如果采用的是相对位置向量，其实也就是用$i-j$去查相应的向量，然后加进模型中，就像`transformer-XL`的做法一样。

因为加入了位置向量，所以要迭代的东西还是不变，首先还是保证了每次只能够看到当前位置以及之前的位置，即下三角矩阵；其次使用多层`Transformer Block`，通过`LayerNorm`与`skip connection`，使得梯度回传更容易（这也是`Pre-Norm`全局残差链接的优点）。当然，代价是效果因为全局残差链接导致神经网络实际深度有所欠缺，效果变差。

![Pre-Norm和Post-Norm](http://images.sakebow.cn/interview/pre-norm-and-post-norm.jpg)

> 图片摘自[Pre-Norm vs. Post-Norm](https://zhuanlan.zhihu.com/p/674704060)
>
> 作者：[@leshare](https://www.zhihu.com/people/leshare-14)

最后，使用交叉熵优化：

$$\mathscr{L}=-\frac{1}{N}\sum_{i=1}\log{p(y_t|x_{i\le t})}$$

# RoPE

旋转位置编码（`Rotary Position Embedding`）的不同之处在于，他不再把位置信息加进向量中，而是在每一层`attention`中，将位置信息按照频率旋转。

`APE`如式$1$，而`RoPE`则要求：

若$d_h$是隐藏层数量、$l=\frac{d_h}{2}-1$、$\omega=10^{4-\frac{2l}{d_h}}$，

且：

$$q_l=\begin{bmatrix}q_{2l}\\q_{2l+1}\end{bmatrix}$$
$$k_l=\begin{bmatrix}k_{2l}\\k_{2l+1}\end{bmatrix}$$

则有：

$$\text{score}=(q_l\cdot k_l)\cdot\cos\omega_l(i-j)+(q_l\times k_l)\cdot\sin\omega_l(i-j)\tag{2}$$

其中，$q_l\cdot k_l$表示内容相关向量矩阵，$q_l\times k_l$表示位置相关向量矩阵。

于是，在`RoPE`中，$\mathbf{Q}$与$\mathbf{K}$的相关性就简化为角度$\omega_l(i-j)$的旋转角度。

在后续的训练中，依然是与`APE`的交叉熵保持一致，只是我们不再需要将位置信息加入到`input_emb`中，只需要考虑旋转位置信息即可。