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
RAG presents a cost-effective alternative to the traditional, resource-intensive training and fine-tuning processes required for LLMs. Unlike standard LLMs that need model updates to incorporate new information, RAG dynamically integrates fresh data by retrieving relevant external content and concatenating it to the search query. This ensures flexibility and scalability to make it particularly valuable across a variety of use cases, like self-service applications where solutions get retrieved from an internal knowledge base and integrated into a helpful answer. Down below you can see the basic workflow that most RAG systems follow. 
</p>

{{< figure align=center alt="Basic Retrieval-Augmented Generation Workflow" src="/imgs/rag/rag_workflow.png" width=100% caption="Figure 2. Basic RAG Workflow []">}}

<p align="justify"> 
You start by generating document vectors using an embedding model for every document or text you want to include in your knowledge database. Embedding models (e.g. BERT, OpenAI text-embedding-ada-002) turn text into a vector representation, which is essential for the retrieval process by enabling similarity search. The documents you would want to retrieve can range from specific company guidelines or documents, a website to solution descriptions to known problems. Popular choices for such vector stores are <a href="https://github.com/facebookresearch/faiss">FAISS</a>, <a href="https://www.trychroma.com/">Chroma</a> or <a href="https://lancedb.com/">Lance</a>.  
</p>

<p align="justify"> 
After you have generated the vector store you are good to go. First your query gets embedded into a vector, like we have described above, which is then used in a similarity search over your vector store to determine the most relevant documents. These documents get retrieved and concatenated to your query before its fed into the LLM. The LLM then produces based on the query and context an answer. This describes the very basic and naive workflow. In practice there are a lot of other things involved, like prompt engineering, a limited context window, limited input tokens of the embedding model, preprocessing of the query and documents, postprocessing of the answer etc. We will touch most of these things in more detail in the next sections. 
</p>

- Open Source Platforms: LangChain, LLamaIndex, DSPy

## Indexing 

Let's start by looking at the indexing process.  
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

## Advanced RAG Pipelines

<p align="justify">
The naive RAG implementation described before is rarely enough to satisfy production grade requirements. This has multiple reasons:  
<ul>
<li><b>Question ambiguity</b>: user questions are not well defined and may lead to irrelevant retrieval results </li>
<li><b>Low retrieval accuracy</b>: retrieved documents may not be equally relevant to the question </li>
<li><b>Limited knowledge</b>: the knowledge base may not include the information the user is looking for </li>
<li><b>Context window performance limitations</b>: trying to "over-retrieve" may hit on the capacity of the context window or otherwise produce a context window that is too big to return a result in a reasonable amount of time </li>
</ul>
Many new RAG patterns have emerged to address these limitations. In the following I will go over some of those techniques, but it is important to note that there is no silver bullet. Each one of these methods may still produce poor results in certain situations or isn't well fitted for your specific use case.
</p>

### Self RAG

<p align="justify">
The Self-Reflective RAG paper describes fine-tuned models that incorporate mechanisms for adaptive information retrieval and self critique. These models can dynamically determine when external information is needed, and can critically evaluate its generated responses for relevance and factual accuracy. Figure X down below shows the retrieval process using these new tokens.
</p>

{{< figure align=center alt="Flowchart of the Self RAG method" src="/imgs/rag/self_rag.png" width=100% caption="Figure X. Self RAG Process [3]">}}

<p align="justify">
First the Language Model decides about the necessity of additional information to answer the prompt. In the second step multiple segments are generated concurrently and each of them is rated in terms of relevance to the prompt and usefulness of the retrieved information. Thereafter each segments is critiqued and the best one is chosen. Then the cycle can repeat, if the model decides to retrieve more information. The table down below shows the four new tokens that make this whole process work.
</p>


<table>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Output</th>
</tr>
    <tr>
        <td>Retrieval Token</td>
        <td>Retrieve additional information from knowledge base</td>
        <td>yes, no</td>
    </tr>
    <tr>
        <td>Relevance Token</td>
        <td>Retrieved information is relevant to the prompt</td>
        <td>relevant, irrelevant</td>
    </tr>
    <tr>
        <td>Support Token</td>
        <td>Generated information is supported by the provided context</td>
        <td>fully supported, partially supported, no support</td>
    </tr>
    <tr>
        <td>Critique Token</td>
        <td>Useful response to the prompt</td>
        <td>5, 4, 3, 2, 1</td>
    </tr>
</table> 

<p align="justify">

</p>


### Corrective RAG 

<p align="justify">

</p>

### RAG Fusion 

<p align="justify">

</p>


## Evaluation


- RAG Finetuning??
- Long Context LLMs  
- Multimodal RAG 
- Advanced RAG Pipelines 



## References 

[[1]](https://arxiv.org/pdf/2404.10981) Y. Huang & J.X. Huang "A Survey on Retrieval-Augmented Text Generation for Large Language Models" (2024).

[[2]](https://www.pinecone.io/learn/advanced-rag-techniques/) Roie Schwaber-Cohen "Advanced RAG Techniques" (2024).

[[3]](https://arxiv.org/pdf/2310.11511) Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2023)