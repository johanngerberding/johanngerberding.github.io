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
tags: ["rag", "transformer", "llm", "vector-database", "retrieval"]
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

- old school algorithms like BM25 focus on term frequency and presence for document ranking and they don't use semantic information of the queries 

- this has changed quite a bit with deep learning 
- modern retrieval strategies leverage pretrained LMs like BERT (Devlin et al. 2019), which capture the semantic meaning of queries more effectively 
- improved search accuracy by considering synonyms and the structure of phrases
- how: measure vector distances between entries in knowledge database (e.g. some vector database) and queries

### Pre-Retrieval

The pre-retrieval phase is all about the data or the knowledge base. The goal is to ensure efficient and correct information retrieval.

#### Data Preparation & Modification

<p align="justify">
If you have worked on an applied ML project with real customers than you are aware of the quality of company data. It would be pretty naive to assume all the information you want to have in your knowledge base is in an easy to read and parse format. Let's be real, most of the company knowledge is in Excel sheets and PowerPoint presentations or in pdf scans that have no pretty text layers. Most of the current RAG pipelines just work on text data, so you have to think about ways to transform everything into text. In the long run, I think you will have multimodal pipelines that can also work with images for example so you can embed e.g. a presentation as a sequence of images. Most of the data preparation phase is about OCR (which still is far from being solved), data cleaning and being creative about how to incorporate all the knowledge you have. You can use frameworks like <a href="https://github.com/Unstructured-IO/unstructured">unstructured</a> for parsing your data. This is really helpful for quickly getting a prototype up and running. But if you have ever seen business Excel sheets you might know why I think that those tools are definitely not good enough and we will need something that works on images.
</p>

<p align="justify">
Data Modification is another important component of the pre-retrieval phase for enhancing retrieval efficiency. It includes e.g. removing irrelevant or redundant information from your text to improve the quality of retrieval results or enriching the data with additional information such as metadata to boost relevance and diversity of the retrieved content. You could e.g. use visual models to describe images and diagrams in text form or use a LLM to extract metadata from documents.
</p>


#### Indexing

<p align="justify">
After our data is clean and pretty we follow up with indexing which is dependent on the task and data type. In the classical workflow, documents/texts are split into chunks and then each chunk is embedded and indexed. The embeddings are created through the use of an embedding model like the one from <a href="https://docs.mistral.ai/capabilities/embeddings/">Mistral</a>. Such an embedding model turns your text into a feature vector that captures the semantic meaning of chunks. Unfortunately most of these embedding models are closed sourced but feel free to build your own.</p>

{{< figure align=center alt="Basic Chunking/Indexing Workflow" src="/imgs/rag/indexing.png" width=80% caption="Figure 3. Basic Chunking/Indexing Workflow">}}

<p align="justify">
As you might already know, LLMs have limited capacity to process text or more specifically tokens, which is often described as context length. E.g. GPT3.5 has a context length of 4,096 tokens which it can process at once. This roughly about 6 pages of english text after tokenization. By chunking our documents we first of all have to make sure, that the chunk size is smaller than the maximum context length of our LLM of choice. Nowadays there exist models like Llama3.1 that have extensive context lengths like 128K tokens. Some people might argue that RAG will become irrelevant in the long term because we can shove our whole knowledge base in-context but that's for the future. In most cases we just need small passages of text from a document and we want to combine it with information from other documents as well to answer our question. Determining the chunk size is a critical decision that not only influences the accuracy but also the efficiency (response generation time) of the retrieval and the overall RAG system. The goal is to choose the correct chunks of optimal lengths so that you can give the LLM as much relevant information as possible.</p>

<p align="justify">
There are a bunch of different possibilities of how to chunk you documents, e.g. page-level, sentence-level or you try to split by sections. You can imagine, based on the use case and data type, different strategies could be helpful here. There is no one size fits all approach and most of the time you have to do a bit of experimenting. E.g. Sentence-level indexing is beneficial for question-answering systems to precisely locate answers, while document-level indexing is more appropriate for summarizing documents to understand their main concepts and ideas. In some cases the retrieved data doesn't even have to be the same we used while indexing. For example you could use a LLM to generate questions that every chunk answers and then you retrieve based on similarity of questions. Or you could generate summaries for every chunk and index those which can reduce redundant information and irrelevant details (Figure 4).</p>

{{< figure align=center alt="Advanced Chunking/Indexing Workflow" src="/imgs/rag/advanced_indexing.png" width=90% caption="Figure 4. Advanced Chunking/Indexing Workflow">}}


<p align="justify">
Both strategies shown above are just two examples of how to improve the chunk indexing and you can be creative here, based on you use case. Another interesting way of indexing is described in MemWalker paper <a href="#references">[6]</a>, where the documents are split into segments which are recursively summarized and combined to form a memory tree. To generate answers the tree is navigated down to the most important segments. The method outperforms various long context LLMs.
</p>


#### Query Manipulation

<p align="justify">
After the indexing is done the last possibility to improve you retrieval pipeline is through query manipulation to adjust the user queries for a better match with the indexed data. Examples of this include:
</p>
<ul>
<li><b>Query Reformulation/Refinement</b>: Rewrite the query to align more closely with the user's 'real' intention (this is hard to do) or create multiple queries with the same intention.</li>
<li><b>Query Expansion</b>: Extend the query to capture more relevant results through e.g. including synonyms or related terms or identifying people/places/companies and add information regarding those.</li>
<li><b>Query Normalization</b>: Resolve differences in spelling or terminology for more consistent query matching.</li>
</ul>
<p align="justify">
This is another part of the pipeline where you can be creative and experiment with different approaches, combine them or modify them for your specific use case. Here is a short list of papers that implemented those strategies or combinations of them: Rewrite-Retrieve-Read <a href="#references">[11]</a>, FiD (Fusion-in-Decoder) <a href="#references">[7]</a>, CoK (Chain-of-Knowledge) <a href="#references">[8]</a>, Query2Doc <a href="#references">[10]</a>, Step-Back <a href="#references">[9]</a> or FLARE (Forward-Looking Active Retrieval) <a href="#references">[12]</a>.
</p>

### Retrieval 

<p align="justify">
The retrieval process is a combination of search and ranking algorithms. The overall goal is to select and prioritize documents from a knowledge base to enhance the quality of the generation model outputs. The first thing you have to think about here is the type of search strategy you want to use. There are three main choices you can select from, vector search, keyword search or a hybrid approach. 

<p align="justify">
Since you have embedded all your knowledge into feature vectors it's kinda easy to just use cosine similarity and k-Nearest-Neighbors to identify the 'k' vectors that are most similar to the query. kNN is a very popular choice due to its simplicity and explainability. One shortcoming of this strategy is the computational intensity of this approach when you have a very large knowledge base. Instead of cosine similarity you can also use euclidean distance or the dot prodoct which can lead to better search results in some cases.

</p>
- BM25 is the most popular keyword retrieval function that is based on bag-of-words
- extensions: BM25F which adds different weights to fields of a document and BM25+ which addresses a problem with the unfair scoring of longer documents

<p align="justify">

</p>



- what do you do when you have your search results?
- reciprocal rank fusion for hybrid search?
<p align="justify">
</p>

### Post-Retrieval

<p align="justify">
</p>

#### Re-Ranking

<p align="justify">
</p>

#### Filtering 

<p align="justify">
</p>



### Generation 

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
</p>  
<ul>
<li><b>Question ambiguity</b>: user questions are not well defined and may lead to irrelevant retrieval results </li>
<li><b>Low retrieval accuracy</b>: retrieved documents may not be equally relevant to the question </li>
<li><b>Limited knowledge</b>: the knowledge base may not include the information the user is looking for </li>
<li><b>Context window performance limitations</b>: trying to "over-retrieve" may hit on the capacity of the context window or otherwise produce a context window that is too big to return a result in a reasonable amount of time </li>
</ul>
<p align="justify">
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
- Why is all this a good idea? (Benchmarks) 
- What are challenges?  
For more detailed information regarding training, how they inserted these special tokens and data collection I'll recommend checking out the <a href="https://arxiv.org/pdf/2310.11511">paper</a>.
</p>


### Corrective RAG 

<p align="justify">
The Corrective Retrieval Augmented Generation method to improve the accuracy and robustness of LM by re-incorporating information from retrieved documents. This is accomplished by the introduction of three additional components. First a lightweight evaluator model that assesses the quality of retrieved documents and determines which to use, ignore or request more documents based on a score. The second new component is an incorporated web search to extend the information base and get more up-to-date information. The last new addition is a decompose-then-recompose algorithm for selectively focusing on key information and filtering out irrelevant information that also makes use of the evaluator model. The Figure X down below gives a nice overview of the CRAG method at inference.
</p>

{{< figure align=center alt="An overview of Corrective Retrieval Augmented Generation" src="/imgs/rag/crag.png" width=100% caption="Figure X. An overview of CRAG at inference [4]">}}

<p align="justify">
The Knowledge Refinement process is quite straightforward and starts by segmenting each retrieved document into fine-grained knowledge strips through heuristic rules. Then the retrieval evaluator is used to calculate a relevance score of each knowledge strip. Based on these scores the strips are filtered and recomposed via concatenation in order.
</p>

### RAG Fusion 

<p align="justify">

</p>


### Self-Reasoning  

<p align="justify">
The method described in this paper focuses on the problem of irrelevant document retrieval which may result in unhelpful response generation or even performance deterioration. It is similar to the Self-RAG without the need of extra models and datasets. The authors propose an end-to-end self-reasoning framework to improve on these problems and especially improve reliability and traceability. 
</p>

{{< figure align=center alt="Self-Reasoning Framework to Improve RAG" src="/imgs/rag/self-reasoning.png" width=100% caption="Figure X. Self-Reasoning Framework to Improve RAG [5]">}}

<p align="justify">
As shown above the framework consists of three processes, the Relevance-Aware Process, the Evidence-Aware Selective Process and the Trajectory Analysis Process.
</p>

<p align="justify">
<b>Relevance-Aware Process</b>: Instruct the model to judge the relevance between the retrieved documents $\mathcal{D}$ and the given question $\mathcal{q}$. In addition the model is requested to generate reasons explaining the relevance of each document. The self-reasoning trajectories are defined as $\mathcal{\tau_{r}}$. If none of the documents is classified as relevant, the answer should be generated by the basic LLM. 
</p>

<p align="justify">
<b>Evidence-Aware Selective Process</b>: Instruct the model to choose relevant documents and select snippets of key sentences for the selected documents. Then the model has to generate the reason why the selected snippets can answer the question, so that you end up with a list containing the cited content and the reason for the cite. The self-reasoning trajectories generated in this process are defined as $\mathcal{\tau_{e}}$.</p>

<p align="justify">
<b>Trajectory Analysis Process</b>: Here the self-reasoning trajectories ($\mathcal{\tau_{r}}$ & $\mathcal{\tau_{e}}$) are consolidated to form a chain of reasoning snippets. With these snippets as context the LLM is instructed to output content with two fields: analysis (long-form answer) and answer (short-form). 
</p>

<p align="justify">
Regarding the training process the authors propose a stage-wise training because of problems with the generation of long reasoning trajectories. In the first stage only the first two stage trajectories are masked, in the second stage only the trajectories from the third process are masked. Finally all trajectories are concatenated and trained end-to-end. For more details on this procedure, check out the paper.</p>


## Evaluation

<p align="justify">
Now you have built you fancy RAG engine, but how do you determine if this thing is any good? Is it even better as a vanilla LLM or is it even worse? Since RAG systems are comprised of multiple components and can get kinda complicated as we have seen above, you can evaluate single components individually or just the performance of the whole thing. Because of the AI hype and the push of LLMs into prodcution systems, this has become an important area of research. 
</p>

Datasets: 
- TriviaQA
- HotpotQA
- FEVER 
- Natural Questions 
- Wizard of Wikipedia
- T-REX 

two aspects to look at: 
- retrieval  
- generation 

problems: 
how to evaluate pre-retrieval stuff? (create multiple different indexes and compare??)

Huggingface RAG Evaluation: 
- build a synthetic evaluation dataset and use a LLM as a judge to compute the accuracy of the system

## Future of RAG

- obsolete because of giant context LLMs? 
- Multimodal RAG? 

## References 

<a name="references"></a>

[[1]](https://arxiv.org/pdf/2404.10981) Y. Huang & J.X. Huang "A Survey on Retrieval-Augmented Text Generation for Large Language Models" (2024).

[[2]](https://www.pinecone.io/learn/advanced-rag-techniques/) Roie Schwaber-Cohen "Advanced RAG Techniques" (2024).

[[3]](https://arxiv.org/pdf/2310.11511) Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2023)

[[4]](https://arxiv.org/pdf/2401.15884) Yan et al. "Corrective Retrieval Augmented Generation" (2024)

[[5]](https://arxiv.org/pdf/2407.19813) Xia et al. "Improving Retrieval Augmented Language Model with Self-Reasoning" (2024)

[[6]](https://arxiv.org/pdf/2310.05029) Chen et al. "Walking Down the Memory Maze: Beyond Context Limit Through Interactive Reading" (2023)

[[7]](https://arxiv.org/pdf/2007.01282) G. Izacard & E. Grave "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering" (2020)

[[8]](https://arxiv.org/pdf/2305.13269) Xingxuan et al. "Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources" (2024)

[[9]](https://arxiv.org/pdf/2310.06117) Zheng et al. "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models" (2023)

[[10]](https://arxiv.org/pdf/2303.07678) Wang et al. "Query2doc: Query Expansion with Large Language Models" (2023)

[[11]](https://arxiv.org/pdf/2305.14283) Ma et al. "Query Rewriting for Retrieval-Augmented Large Language Models" (2023)

[[12]](https://arxiv.org/pdf/2305.06983) Jiang et al. "Active Retrieval Augmented Generation" (2023)