---
title: Retrieval-Augmented Text Generation (RAG) 
date: 2024-04-23 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: A short intro to Retrieval-Augmented Text Generation, what is it, why is it useful and how does it work. 
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

- data prep is key here! 
- involves: 
    text normalization (tokenization, stemming, removal of stop words)
    organize text segments into sentences or paragraphs to facilitate more focused searches 

- generation of semantic vector representations of texts which are stored in a vector database for rapid and precise retrieval 
- to achieve this you use a so called embedding model 

## Retrieval

- old school algorithms like BM25 focus on term frequency and presence for document ranking and they don't use semantic information of the queries 

- this has changed quite a bit with deep learning 
- modern retrieval strategies leverage pretrained LMs like BERT (Devlin et al. 2019), which capture the semantic meaning of queries more effectively 
- improved search accuracy by considering synonyms and the structure of phrases
- how: measure vector distances between entries in knowledge database (e.g. some vector database) and queries

### Pre-Retrieval

- data and query preparation 
- it includes indexing
- you can imagine, based on the use case and data type, different strategies could be helpful here
- e.g. sentence-level indexing is beneficial for question-answering systems to precisely locate answers, while document-level indexing is more appropriate for summarizing documents to understand their main concepts and ideas 
- the next thing you have to think about is query manipulation, which is performed to adjust user queries for a better match with the indexed data
- query reformulation: rewrite the query to align more closely with the user's intention 
- query expansion: extend the query to capture more relevant results through synonyms or related terms 
- query normalization: resolve differences in spelling or terminology for consistent query matching
- another way to enhance retrieval efficiency is data modification
- this includes preprocessing techniques like removing irrelevant or redundant information or enriching the data with additional information such as metadata to boost the relevance and diversity of the retrieved content

### Retrieval 

<p align="justify">
- combination of search and ranking 
- goal: select and prioritize documents from a knowledge base to enhance the quality of the generation models outputs 
- distance based search in a vector database (cosine similarity)
</p>

### Post-Retrieval

<p align="justify">
</p>

## Generation 

<p align="justify">
- task: generate text that is relevant to the query and reflective of the information found in the retrieved documents
- how: concat the query with the found information and input that into a LLM
- challenge: ensuring the outputs alignment and accuracy with the retrieved contents isn't straightforward 
- the generated output should accurately convey the information from the retrieved documents and align with the query's intent, while also offering the flexibility to introduce new insights or perspectives not explicitly contained within the retrieved data 
- another challenge: what if there is no relevant information in your knowledge base for a certain query, how to detect such instances and how to behave? 
- retrieval is kinda naive, how would you filter irrelevant documents? based on some distance metric, but how to define that distance? 
</p>

## Advanced RAG Pipeline

<p align="justify">

</p>

## Evaluation


- RAG Finetuning??
- Long Context LLMs  
- Multimodal RAG 
- Advanced RAG Pipelines 



## References 


