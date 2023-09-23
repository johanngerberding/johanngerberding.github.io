---
title: Parameter-Efficient LLM Finetuning  
date: 2023-09-08 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: Finetuning of Large Language Models in a parameter efficient way.
include_toc: true
showToc: true
math: true
draft: true
---

## Introduction

I am already quite late to the Large Language Models (LLM) party but better starting late than never. In this post I am going over a couple of popular techniques for fine-tuning these models without the need of training the whole thing which would be quite expensive. But before I dive into the techniques we should talk about what LLMs and fine-tuning are, why we need it and what the current problems are.

LLMs are based on the [Transformer]() architecture, like the GPTs or BERT, which have achieved state-of-the-art results in various Natural Language Processing (NLP) tasks, like Translation, Question Answering, Summarization etc. The paradigm these days is to train such a model on generic web-scale data and fine-tune it on a downstream task. Fine-tuning in this case just means that you train the model further, but for the downstream task you need way less data. This fine-tuning results in most cases in huge performance gains when compared to using just the LLM as is (e.g. zero-shot inference).  


- why do we need to fine-tune these models: don't perform well on more narrow or specific tasks  
- what are the problems -> training all parameters is very expensive 

huggingface provides a nice repo containing a lot of PEFT methods for you: https://github.com/huggingface/peft  

In the following I will going to dive a bit deeper in how some of these methods work.

What is finetuning? Why do we need it? What are the current problems? 
<p align="justify">
- in-context learning -> models are able to perform tasks by providing them the right context (examples of your task) -> this is cool when you only have access to the model via an API or UI (this only works with generative models like the GPTs) 
- this works fine in some cases but for more specific things you still have to gather a dataset for the task and the specific domain and then finetune that model, this would result in superior results than just in-context learning (I don't know why they call this learning)
- what does finetuning even mean here? -> it basically means that you want to train the model on your task specific dataset 
- there are two ways to do this: 
1. the feature based approach, where you keep the transformer weights frozen (no training here) and just train a classifier which you feed the output embeddings of the frozen LLM (e.g. logistic regression model, random forest or XGBoost)
2. finetuning, where you train the whole model or a bigger portion of it (e.g. your just train the output layers after the Transformer blocks) 
- in general training the whole model will give you the best results but it is also the most expensive one    
- the problem is, that the currently best performing models are so big that there is no way that you can finetune this thing on your personal computer anymore
- so the question of how to utilize them more efficiently and effectively has become a very active area 
- people want to run those models on their laptops or desktop pcs without buying multiple graphics cards 
- so what do you do when in-context learning doesnt cut it for your use case and you don't have the compute to finetune those bigger models?
- researchers developed several techniques to make the finetuning of LLMs much more efficient by training just a small number of parameters
- I try to explain some of the most popular parameter-efficient finetuning techniques (PEFT) in this blogpost: Prefix Tuning, Adapters, Low Rank Adaptation (LoRA)
 
</p>


## Prefix Tuning  

<p align="justify">
- hard prompt tuning -> change the discrete input tokens, e.g. provide examples what the model should do, non-differentiable 
- soft prompt tuning -> concat embeddings of input tokens with a trainable tensor that you can backpropagate through 
- prefix tuning is a form of soft prompt tuning which keeps model parameters frozen and optimizes a small continuous task-specific vector called prefix
- idea: add a trainable tensor to each transformer block (instead of only the input embeddings)


</p>

## Adapters 

<p align="justify">

</p>


## Low Rank Adaptation (LoRA)

<p align="justify">
LoRA and QLoRA 
</p>


## Conclusion 

<p align="justify">
</p>


## References 

Sebastian Raschka - Understanding Parameter Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters (2023) 
https://lightning.ai/pages/community/article/understanding-llama-adapters/

Sebastian Raschka - Parameter-Efficient LLM Finetuning With Low Rank Adaptation (2023) 
https://lightning.ai/pages/community/tutorial/lora-llm/

Edward Hu et. al - LoRA: Low-Rank Adaptation of Large Language Models (2021) 
https://arxiv.org/pdf/2106.09685.pdf

Renrui Zhang et. al - LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention (2023)
https://arxiv.org/pdf/2303.16199.pdf

Xiang Lisa Li - Prefix-Tuning: Optimizing Continuous Prompts for Generation (2021) 
https://arxiv.org/pdf/2101.00190.pdf

Tim Dettmers et. al - QLORA: Efficient Finetuning of Quantized LLMs (2023)
https://arxiv.org/pdf/2305.14314.pdf
