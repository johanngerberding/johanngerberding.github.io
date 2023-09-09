---
title: Parameter-Efficient LLM Finetuning  
date: 2022-09-08 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: Finetuning of Large Language Models in a parameter efficient way.
include_toc: true
showToc: true
math: true
draft: true
---

## Introduction

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
