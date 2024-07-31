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
tags: ["llm", "finetuning", "transformer"]
---

## Introduction

<p align="justify">
I am already quite late to the Large Language Models (LLM) party but better starting late than never. In this post I am going over a couple of popular techniques for fine-tuning these models without the need of training the whole thing which would be quite expensive. But before I dive into the techniques we should talk about what LLMs and fine-tuning are, why we need it and what the current problems are.
</p>

<p align="justify">
LLMs are based on the <a href="http://jalammar.github.io/illustrated-transformer/">Transformer</a> architecture, like the GPTs or BERT, which have achieved state-of-the-art results in various Natural Language Processing (NLP) tasks, like Translation, Question Answering, Summarization etc. The paradigm these days is to train such a model on generic web-scale data (basically the whole internet) and fine-tune it on a downstream task. Fine-tuning in this case just means that you train the model further with a small dataset you collected for a specific task. This fine-tuning results in most cases in huge performance gains when compared to using just the LLM as is (e.g. zero-shot inference).  
</p>

<p align="justify">
However, based on the current model sizes a full fine-tuning becomes infeasible to train on consumer hardware (which makes me a bit sad). In addition when you want to store and deploy multiple fine-tuning model instances for different tasks, this becomes very expensive because they are the same size as the base LLM. Because of these two main problems, people came up with more efficient methods for doing this which are referred to as Parameter-efficient fine-tuning (PEFT) procedures. These approaches basically enable you to get performance comparable to full fine-tuning while only having a small number of trainable parameters.  
</p>

<p align="justify">
PEFT approaches only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs. They have also been shown to be better than fine-tuning in cases when you don't have that much task data and in out-of-domain scenarios. By saving just the extra model parameters you also solve the portability and cost problem because the PEFT checkpoints are much smaller, just a few MB, in contrast to the LLM checkpoints that need multiple GB. The small trained weights from PEFT approaches are added on top of the pretrained LLM. So the same LLM can be used for multiple tasks by adding small weights without having to replace the entire model. huggingface provides a nice <a href="https://github.com/huggingface/peft">library</a> for a lot of PEFT methods. 
</p>

<p align="justify">
In the following I will going to dive a bit deeper in how some of these methods work. We will cover Prefix Tuning, Adapters and Low Rank Adaptation (LoRA).
</p>


## Prefix Tuning  

<p align="justify">
There exists different forms of prompt tuning, one of them is hard prompt tuning, also called prompt engineering or in context learning, where you change the input to the model, e.g. provide examples for how the model should answer. This form of tuning is non-differentiable, so the weights don't change and you don't add any weights. Prefix tuning is a form of soft prompt tuning where you concatenate a trainable tensor to the embeddings of the input tokens. This tensor is trainable, so you add a small amount of weights. The model parameters are kept frozen and you just optimize a small continuous task-specific vector which is called prefix.  
</p>

{{< figure align=center alt="Finetuning vs. Prefix Tuning" src="/imgs/parameter_efficient_llm_finetuning/finetuning_vs_prefix_tuning.png" width=70% caption="Figure 1. Finetuning vs. Prefix Tuning">}}

<p align="justify">
Prefix tuning shines especially in low-data settings and the extrapolation to new tasks, e.g. summarization of texts with different topics, is better. It has up to 1000x fewer parameters to learn than in a fine-tuning setting. Another very cool thing is, that it enables personalization by having a different prefix per user trained only on the user data (no cross-contamination) and you could do batch processing of multiple users/tasks and one LLM. 
</p>

## Adapters 

<p align="justify">
Adapter methods are somewhat related to prefix tuning as they also add additional trainable parameters to the original LLM. Instead of prepending prefixes to the input embeddings, you add adapter layers into the transformer blocks. As you can see in the figure down below, the fully connected network in the adapter module has a bottleneck structure similar to an autoencoder which keeps the number of added parameters low and makes the method quite efficient.
</p>

{{< figure align=center alt="Adapter Module Layout" src="/imgs/parameter_efficient_llm_finetuning/adapter_module_layout.png" width=70% caption="Figure 2. Adapter Module Layout">}}

<p align="justify">
In the orginal adapter paper, the authors trained a BERT model with this method and reached a modeling performance comparable to a fully finetuned BERT model while only requiring the training of 3.6% of the parameters. Based on the original prefix tuning paper, the performance of adapters matches the performance of prefix tuning of 0.1% of the parameters at 3%, which makes it less efficient.
</p>

<p align="justify">
Llama-Adapter combines the two ideas of prefix tuning and adapters for LLaMA models from Meta. Each transformer block in the model has its own distinct learned prefix, allowing for more tailored adaptation across different model layers. On top of that, LLaMA-Adapter introduces a zero-initialized attention mechanism coupled with gating.
</p>



## Low Rank Adaptation (LoRA)

<p align="justify">
LoRA and QLoRA 
</p>


## Conclusion 

<p align="justify">
</p>

## Notes 
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
 
and Baogao et. al wrote a paper comparing a lot of these methods and introduced HiWi  
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

Parameter-Efficient Fine-Tuning without Introducing New Latency (2023)
https://arxiv.org/pdf/2305.16742.pdf