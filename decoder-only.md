---
title: 面试记录：你是怎么微调千问大模型的？
date: 2025-08-28 14:34:20
categories:
  - interview
tags:
  - LLM
  - interview
mathjax: true
---

# 前言

表面上看，这个是在问实际经验，但是实际上，这里有一个很重要的前提：`Qwen`在一代、二代，甚至之后的$2.5$等版本，都是`decoder-only`的，这也就从本质上改变了输入数据的格式。

<!-- more -->

# 对比差距

在最初的`MHA`结构中，有一个`encoder`和一个`decoder`，而千问从$1$到$3$采用的都是`decoder-only`。

这个`decoder-only`是什么结构？

简单地讲，可以总结为：

![对比](http://images.sakebow.cn/interview/comparsion.png)

> 图片摘自[Deep (Learning) Focus](https://cameronrwolfe.substack.com/)的文章：[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

原本的结构中，存在一个`encoder`，而`decoder-only`则只存在一个`decoder`。

为什么偏偏是`encoder`删掉了？它有什么作用？

# 被删掉的encoder到底有什么作用

我们重新回到`Transformer`的图中：

![Transformer](http://images.sakebow.cn/interview/transformer.png)

不难发现，`transformer`的输入实际上是同时带有`input`和`output`，输出是`output`的概率预测。也就是说，`encoder-decoder`的擅长领域其实并不是单纯的文本生成，毕竟单纯的去预测下一个`token`，完全没必要输入`output`。而需要`input`和`output`同时输入的模型，本质上也是一种拟合模型。

到这里也就基本能猜个大概了：`encoder-decoder`模型，学习的应该是`input`和`output`的对应关系。

所以，对于这一点，`milvus`也给出了自己的答案：

`encoder-decoder`结构在**输入与输出在结构或者意义上存在显著差异的时候表现优异**。

> 摘自：[milvus官网](https://milvus.io/zh)的文章[what-are-decoderonly-models-vs-encoderdecoder-models](https://www.milvus.io/cn/blogs/2022/03/07/how-to-choose-the-right-model-for-your-use-case.html)

为了佐证上述观点，我们再来看一个`encoder-only`：`BERT`。

不难发现，`BERT`的核心是找到段落中文本的正确顺序与词语的正确含义。

所以也有人总结：

- `encoder-only`适合分类任务
- `encoder-decoder`适合翻译任务
- `decoder-only`适合生成任务

也是有一定道理的。

那么，我们再回过头去解释，`decoder-only`删去了原本的`encoder`，取消了分类任务的适配性，直接考虑生成任务，虽然会带来理解能力的退化，但是增强了基于上下文生成的能力。

# 如何微调

那么我们最后再回到最初的问题：怎么样微调。

既然是`decoder-only`，那么其实我们所需要的并不是一个`input`与`output`对照着的问答对，其实连续长文本就足够训练。

其中，在微调的过程中，面对不同场景、不同业务，也需要采取不同的手段。

## Fine-tuning

`Fine-tuning`，翻译过来其实是**全量微调**。在维基百科中，他的解释是：

> 在深度学习中，`fine-tuning`是一种迁移学习，将预训练的神经网络参数在新的数据上再次训练。
>
> ——摘自[维基百科](https://en.wikipedia.org)的[Fine-tuning(deep learning)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning))词条

在这个过程中，全量微调主要面相领域任务，并将所有的参数一次性全部调整到位。比如，对于通信领域大模型，直接喂一整段通信领域相关的连续长文本，即可开始全量微调。对于原始参数权重$\mathbf{W}$，本质上依然是梯度下降：

$$\mathbf{W}^\prime = \mathbf{W} - \alpha\nabla_\theta\frac{\partial L}{\partial\mathbf{W}}\tag{1}$$

## LoRA

`LoRA`，英文全称`Low-Rank Adaptation`，翻译过来是**低秩适配**。这个过程并没有修改所有的矩阵，而是将原始权重$\mathbf{W}$冻结，然后给出两个低秩矩阵$\mathbf{A}$和$\mathbf{B}$，用于模拟参数的更新。于是，更新后的权重就可以表示为：

$$\mathbf{W}^\prime = \mathbf{W} + \alpha\mathbf{B}\mathbf{A}\tag{2}$$

此时，可更新的部分就不再是$\mathbf{W}$，而是$\mathbf{A}$和$\mathbf{B}$。

## Prefix-tuning

`Prefix-tuning`，翻译过来是**前缀微调**。与`RAG`相同的是，他的任务本质是将一段前缀拼接到`prompt`前方；不同的是，`Prefix-tuning`的前缀是一段可学习的向量，在后续过程中，也将采用`MHA`结构对这段可学习的向量进行优化。

为了和其他的学习方式做个区分，可学习的前缀包括$K_p$、$V_p$，并在`MHA`中引入为：

$$\text{Attention}(Q,K,V)=\text{softmax}\left(
\frac{Q[K;K_p]^\top}{\sqrt{d_k}}\right)[V;V^\top]\tag{3}$$

可以看到，它本质上修改的是神经网络结构信息，而不是单纯的在输入数据中添加前缀。

## Prompt-tuning

要说在输入数据前添加前缀，这个就是最匹配的了。具体而言，就是直接在`prompt`的`embedding`前面再拼上一段前缀，构成一段新的`input_emb`。

顺带一提，他还有个名字是`P-tuning`。

为了与式$3$做个区分，这里本质上并没有对`MHA`进行修改，而是在原先的基础上，额外添加一份可学习的前缀向量。

我们知道，`MHA`主要提升了大模型的模型推理性能，因此，`P-tuning`的思路，本质上是为了提升模型的零样本分类能力，也就是优化`prompt`泛用性。