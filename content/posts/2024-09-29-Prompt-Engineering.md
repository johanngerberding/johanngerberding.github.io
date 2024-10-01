---
title: Prompt Engineering Techniques
date: 2024-09-29 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: Prompt Engineering is a thing now, for real. 
include_toc: true
showToc: true
math: true
draft: true 
tags: ["vlm", "llm", "prompt engineering"]
---


## Introduction  

- lets make sure you do not fall for the prompt engineering scammer bullshit on twitter, believe me, you don't need a course for that, I hope I can save you a lot of money with this one
- i am late to this as always 

<p align="justify">
</p>


## In-Context Prompting 

- zero-shot learning (prompting) => just ask the model 
```yaml
Prompt:: I think this blog is absolutely awesome.
Sentiment::  
```
- few-shot learning (prompting) => present a set of high-quality examples of input and desired output, which often leads to better performance than zero-shot 
- but you need more tokens so watch out for your context length   
- observations of studies that looked into few-shot prompting to maximize performance: choice of prompt format, training examples and the order of examples can lead to dramatically different performance from near random to state-of-the-art (which is not good) 
- how to select examples: 
    - choose examples that are semantically similar to the test example using k-NN clustering in the embedding space 
    - Su et al. (2022): how to minimize annotation cost for in-context examples -> propose a general framework constisting of two steps, selective annotation and prompt retrieval (embed examples using Sentence-BERT and then calculate cosine similarity); selective annotation method called Vote-k: 
<p align="justify">
</p>


## Instruction Prompting

<p align="justify">
</p>

## Chain-of-Thought

<p align="justify">
</p>

- maybe add multimodal CoT here for VLMs 
- https://arxiv.org/pdf/2302.00923


## Self-Consistency Sampling

<p align="justify">
</p>

## Tree-of-Thought 

<p align="justify">
</p>

## Automatic Prompt Design

<p align="justify">
</p>

## Augmented Language Models

<p align="justify">
</p>

## References 

<a name="references"></a>

[[1]](https://www.promptingguide.ai/) Elvis Saravia "Prompt Engineering Guide" (2022).

[[2]](https://arxiv.org/pdf/2201.11903) Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022).

[[3]](https://arxiv.org/pdf/2203.11171) Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2023).

[[4]](https://arxiv.org/pdf/2305.10601) Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023).