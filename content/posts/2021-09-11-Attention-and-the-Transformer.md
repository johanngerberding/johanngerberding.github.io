---
layout: post
title: Attention and the Transformer
date: 2021-10-15 10:45:16 +0200
author: Johann Gerberding
summary: Introduction to the world of Reinforcement Learning, where I cover the basics and some algorithms.
include_toc: true
showToc: true
math: true
draft: true
---


## Whats wrong with RNN's?

<p align="justify">
Before we dive into the details of the (Vanilla) Transformer model architecture I want to give you a short intro about how the self-attention mechanism, which is one of the key elements of a transformer block, evolved and why it is part of so many state-of-the art approaches, especially in Natural Language Processing (In the meantime, it can be said that they are also gradually taking over the computer vision field; the state of the art in the field of image classification is a combination of a convolutional neural net and a transformer, called CoAtNet[9]). Much of this information comes from the 2019 lecture by [Justin Johnson](https://www.youtube.com/watch?v=YAgjfMR9R_M&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=14&ab_channel=MichiganOnline) (Michigan State University) and the blogpost [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) from Lilian Weng, which I think are two of the best resources for getting started on the topic.
</p>

<p align="justify">
Let's start by looking at pre-Transformer sequence-to-sequence architectures. Classic seq2seq models have an encoder-decoder architecture and aim to transform an input sequence (source, e.g. german sentence) into a different output sequence (target, e.g. english translation). Both sequences can be of arbitrary length and the Encoder as well as the Decoder are different Recurrent Neural Network architectures (e.g. LSTM, GRU).
</p>

![Encoder Decoder Architecture]({{ '/assets/imgs/transformer/encoder_decoder.png' | relative_url}}){: style="width: 100%; margin: 0 auto; display: block;"}**Figure 1.** Encoder Decoder Architecture [5]

<p align="justify">
Typical transformation tasks of those kinds of models are f.e. machine translation, question-answer dialog generation, image/video captioning [7], speech recognition [6] or parsing sentences into grammar trees [8]. The encoder and the decoder network are typically connected with a fixed length context vector which transfers information between the encoded and the decoded sequence but becomes a bottleneck for longer sequences because of its fixed size. Often the model has "forgotten" the first part of a long sequence once it completes processing the whole input.
</p>

<p align="justify">
To solve this problem, the **Attention mechanism** was born. Instead of only relying on the last hidden state, the idea was to create shortcut connections between the context vector and the entire source input which should negate the "forgetting". The alignment between the source and the target sequence is learned and controlled by the context vector. The illustration down below shows this mechanism.
</p>

![Additive Attention Mechanism]({{ '/assets/imgs/transformer/additive_attention.png' | relative_url}}){: style="width: 100%; margin: 0 auto; display: block;"}**Figure 2.** Additive Attention Mechanism used in [5] from [10]

<p align="justify">
To better understand how this works I have re-implemented this [here](). Feel free to clone the repo and train the model yourself. The Attention mechanism used is called **Additive Attention** (there exist different forms of Attention mechanisms). The encoder consists of a bidirectional RNN with a forward and a backward hidden state, $\overrightarrow{\boldsymbol{h_{i}}}$ and $\overleftarrow{\boldsymbol{h_{i}}}$, which are concatenated to form the encoder state $\boldsymbol{h_{i}}$. The context vector $\boldsymbol{c_{t}}$ for the output $y_{t}$ is a sum of hidden states of the input sequence weighted by alignment scores where $n$ is the length of the input sequence:
</p>
<p align="center">
$$
\boldsymbol{c}_t = \sum_{i=1}^{n}\alpha_{t,i} \boldsymbol{h}_{i} \\
\alpha_{t,i} = softmax(score(\boldsymbol{s}_{t}, \boldsymbol{h}_{i})) \\
score(\boldsymbol{s}_{t}, \boldsymbol{h}_{i}) = \boldsymbol{v}_{a}^{T} tanh(\boldsymbol{W}_{a}[\boldsymbol{s}_{t}; \boldsymbol{h}_{i}])
$$
</p>
<p align="justify">
The alignment model in [5] is a feed-forward network with a single hidden layer. It assigns a score to the pair of input at position $i$ and output at position $t$ based on how well the two words match. Both $\boldsymbol{v_{a}}$ and $\boldsymbol{W_{a}}$ are learned by the alignment model. Based on this, the decoder network calculates the hidden state:
</p>
<p align="center">
$$
\boldsymbol{s}_{t} = f(\boldsymbol{s}_{t-1}, y_{t-1}, \boldsymbol{c}_{t}) \\
$$
</p>
<p align="justify">
With the alignment scores you can create pretty cool matrices which show the correlation between the source and the target words. Down below I have created such a plot with a model I trained for a few epochs on the Multi30k torchtext dataset.
</p>

![Matrix of alignment scores]({{ '/assets/imgs/transformer/attention_matrix.png' | relative_url}}){: style="width: 80%; margin: 0 auto; display: block;"}**Figure 3.** Attention score matrix

<p align="justify">
If you want to learn more about different forms of Attention and popular alignment score functions you can take a look at [10] which provides a table summarizing this.
</p>

## (Self)-Attention

- at each timestep of decoder, the context vector "looks at" different parts of the input sequence
- how: calculate attention scalars based on the hidden state and the decoder state for every hidden state, then multiply the hidden states by the attention scalars and sum them up to create a context vector

this mechanism doesn't use the fact that hidden state i forms an ordered sequence, it just treats the hidden states as an unordered set $\{h_{i}\}$ 

this means that you can use a similar architecture given any set of input hidden vectors (e.g in image captioning)

in CS, if we discover something useful which is generally applicable we try to abstract


![Self-Attention Layer]({{ '/assets/imgs/transformer/self_attention_layer.png' | relative_url}}){: style="width: 100%; margin: 0 auto; display: block;"}**Figure 2.** Self-Attention Layer ([Michigan Online - Justin Johnson](https://www.youtube.com/watch?v=YAgjfMR9R_M&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=14&ab_channel=MichiganOnline)) 


"Memory is attention through time" - Alex Graves (2020)

Attention is about ignoring things to focus on specific parts of the data.

## Transformer

<p align="justify">
I think it is fair to say that the **Transformer** is by far the most popular model architecture choice in the research community at the moment. Vaswani et al. presented the architecture in their paper titeled "Attention is All you Need" which already gives an idea of what it is all about. The Transformer is entirely built on the self-attention mechanism presented before without using any sequence aligned recurrent architecture.
</p>

### Architecture

**Encoder**


**Decoder**


### Positional Encoding


### Multi-Head Attention


**Scaled Dot-Product Attention**

$$
Attention(Q,K,V) = softmax(QK^{T}/)V
$$

**Multi-Head Attention**


## Applications

### NLP

a few popular NLP architectures (GPT, Bert and stuff)

### Computer  Vision

Vision Transformer

## Summary


## References

[[1]](https://arxiv.org/pdf/1706.03762.pdf) Vaswani et al. "Attention is all you need", 2017.

[[2]](http://peterbloem.nl/blog/transformers) Transformers from Scratch.

[[3]](https://theaisummer.com/attention/) How Attention works in Deep Learning: Understanding the attention mechanism in sequence models.

[[4]](https://www.youtube.com/watch?v=YAgjfMR9R_M&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=14&ab_channel=MichiganOnline) Lecture 13: Attention, Justin Johnson.

[[5]](https://arxiv.org/pdf/1409.0473.pdf) Bahdanau et al. "Neural Machine Translation by jointly learning to align and translate", 2016.

[[6]](https://arxiv.org/pdf/1610.03022.pdf) Zhang et al. "Very Deep Convolutional Networks for End-to-End Speech Recognition", 2016.

[[7]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7505636) Vinyals et al. "Show and Tell: Lessons Learned from the 2015 MSCOCO Image Captioning Challenge", 2017.

[[8]](https://proceedings.neurips.cc/paper/2015/file/277281aada22045c03945dcb2ca6f2ec-Paper.pdf) Vinyals et al. "Grammar as a Foreign Language", 2015.

[[9]](https://arxiv.org/pdf/2106.04803v2.pdf) Dai et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes", 2021.

[[10]](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) Lilian Weng "Attention? Attention!", 2018.