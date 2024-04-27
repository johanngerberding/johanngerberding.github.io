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

<p align="justify">
In 2020, Lewis et al. introduced a novel approach to enhance language model outputs through the use of Retrieval-Augmented Generation (RAG). Since its inception, RAG has evolved significantly, providing a robust solution to some of the persistent challenges faced by Large Language Models (LLMs).
</p>

<p align="justify">
LLMs, while transformative, are not without their issues. They often produce "hallucinations" or convincing yet inaccurate answers, due to their sole reliance on the fixed dataset on which they were trained on. This limitation is compounded by their inability to incorporate information that emerges after their last training update, rendering them outdated as the world evolves. Additionally, their performance can be suboptimal in specialized areas because their training prioritizes breadth over depth to ensure broad accessibility and applicability.
</p>

<p align="justify">
RAG addresses these challenges by dynamically supplementing model responses with external data fetched in real time in response to queries. This approach not only keeps the information current but also enhances the accuracy and relevancy of the responses.
</p>

<p align="justify">
For a RAG system to deliver high-quality outputs, two core components are crucial. First, it must effectively retrieve the most relevant information from external sources. This involves a sophisticated understanding of the query context and the ability to sift through vast amounts of data quickly and efficiently. Second, it needs to generate accurate responses that integrate the retrieved information, ensuring that the final output is both coherent and contextually appropriate.
</p>

<p align="justify">
Building on these technologies, startups <a href="https://www.trychroma.com/">like Chroma</a>, <a href="https://weaviate.io/">Weaviate</a>, and <a href="https://www.pinecone.io/">Pinecone</a> have expanded upon RAG's foundational concepts by incorporating robust text storage solutions and advanced tooling into their platforms. These enhancements facilitate more efficient data handling and retrieval processes, which are critical for delivering precise and contextually relevant responses.
</p>

<p align="justify">
In the upcoming sections, we will dig deeper into the world of RAG by providing a comprehensive overview of the framework and highlighting the key components essential for an effective RAG system. Additionally, we will explore practical strategies that can enhance the quality and performance of your RAG system. While RAG offers significant benefits, it is not without its challenges. Therefore, we will also examine some of the common obstacles encountered in building RAG systems and discuss the current limitations that engineers need to be aware of as they build these kind of systems.
</p>


## RAG Framework 

<p align="justify">
The basic workflow of a RAG system consists of three fundamental steps: indexing, retrieval, and generation. This sequence begins with indexing, where data is organized to facilitate quick and efficient access. The retrieval step can be further divided into three subparts: pre-retrieval, retrieval, and post-retrieval. Pre-retrieval involves preparing and setting up the necessary parameters for the search. Retrieval is the actual process of fetching the most relevant information based on the query. Post-retrieval includes the refinement and selection of the fetched data to ensure its relevance and accuracy. Finally, in the generation phase, the system synthesizes this information into coherent and contextually appropriate responses. These foundational steps remain at the core of every advanced RAG workflow, serving as the building blocks for more complex systems.
</p>

{{< figure align=center alt="Abstract Concept Graphic Retrieval-Augmented Generation Elements" src="/imgs/rag/rag_framework.png" width=100% caption="Figure 1. RAG Framework Paradigm [1]">}}

<p align="justify">
RAG presents a cost-effective alternative to the traditional, resource-intensive training and fine-tuning processes required for LLMs. Unlike standard LLMs that necessitate extensive data ingestion and model updates to incorporate new information, RAG dynamically integrates fresh data by retrieving relevant external content during the generation process. This capability allows RAG to remain current and adapt to new data without the need for continuous, deep integration into its core model structure. The flexibility and scalability of RAG make it particularly valuable across a variety of use cases, enabling it to provide tailored, up-to-date responses that are informed by the most recent content available. This makes RAG not only a more efficient choice in terms of computational and time resources but also a more versatile tool in rapidly evolving domains.
</p>

{{< figure align=center alt="Basic Retrieval-Augmented Generation Workflow" src="/imgs/rag/rag_workflow.png" width=100% caption="Figure 2. Basic RAG Workflow []">}}



- RAG evolved from just providing information to multiple interactions between the retrieval and generation components
- conducting several rounds of retrieval to refine the accuracy of the information retrieved 
- then iteratively improve the quality of the generated output

- Open Source Platforms: LangChain, LLamaIndex, DSPy

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

[[1]](https://arxiv.org/pdf/2404.10981) Y. Huang & J.X. Huang "A Survey on Retrieval-Augmented Text Generation for Large Language Models" (2024).
