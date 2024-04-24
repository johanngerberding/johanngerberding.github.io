---
title: Retrieval-Augmented Text Generation (RAG) 
date: 2024-04-23 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: A short intro to Retrieval-Augmented Text Generation, what is it, why is it important and how does it work. 
include_toc: true
showToc: true
math: true
draft: true
---

## Introduction

- was introduced in 2020 by lewis et al and has advanced quite a lot since then 

LLM problems:
- hallucinations = generate convincing yet inaccurate answers (reliance on training dataset)
- not up-to-date, new information that was generated post-training
- subpar performance in specialized areas (LLMs are trained on broad and general data to maximize accessibility and applicability)

RAG supplementing models by fetching external data in response to queries, thus ensuring more accurate and current outputs 

central to high quality RAG:
- effectively retrieve relevant information
- generate accurate responses 



## RAG Framework 

- basic RAG workflow 
1. indexing 
2. retrieval 
3. generation 

- this is still the basis for every more advanced RAG workflow

- retrieval can further split up in 3 subparts:
    1. pre-retrieval 
    2. retrieval 
    3. post-retrieval

- RAG is a more cost effective alternative to the extensive training and fine-tuning processes typically required for LLMs.
- dynamic incorporation of fresh information without integrating it into the LLM 
- this makes RAG flexible and scalable 
- useful for diverse use cases 

- RAG evolved from just providing information to multiple interactions between the retrieval and generation components
- conducting several rounds of retrieval to refine the accuracy of the information retrieved 
- then iteratively improve the quality of the generated output

- Platforms: LangChain, LLamaIndex, DSPy

## Indexing 

- the basic rag workflow starts with the creation of an index comprising external sources (e.g. Websites, company relevant pdfs)
- basis for retrieval of relevant information 


## Retrieval


### Pre-Retrieval


### Retrieval 



### Post-Retrieval



## Generation 



## Evaluation


- RAG Finetuning??
- Long Context LLMs  
- Multimodal RAG 
- Advanced RAG Pipelines 



## References 


