---
title: Vision Language Models 
date: 2024-08-23 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: LLMs are boring but VLMs are awesome, let's see why. 
include_toc: true
showToc: true
math: true
draft: true 
tags: ["vlm", "llm"]
---


## Notes 

### Vision Language Models Explained 
[1]

What is a VLM? 
* multimodal models that can learn from images and text 
* generative models
    - input = image, text 
    - output = text
* use cases: image chatting, image recognition, visual question answering, document understanding, image captioning and further more
* some of them are able to perform object detection, segmentation or reasoning about relative positions of objects (which is kinda fire)

### An Introduction to Vision-Language Modeling 
[2]

* Contrastive Method 



- SigLIP (2023)
- Llip (2024) 

* Masking Method 

- FLAVA (2022) 
- MaskVLM (2023)

* Generative-based VLMs 

- CoCa (2022)
- CM3Leon (2023)

* VLMs from Pretrained Backbones 

- Qwen-VL(-Chat) (2023)
- BLIP2 (2023)



## Introduction  

I will not talk about closed source models because they are not interesting and nobody likes the closed stuff anyway (and there is no info, so nothing to talk about)


* How do they work? 
* Why are they cool and important?
* What are cool open source models? 
    - InternVL(2)
    - LlaVa
    - Flamingo
    - CoCa
    - BLiP2 (Salesforce)

* current problems of VLMs: 
    - understanding spatial relationships
    - counting stuff (without complicated engineering overhead that relies on additional data annotation)
    - lack understanding of attributes and ordering
    - ignorance of parts of the input prompt (a lot of prompt engineering is needed to produce the results you want)
    - classic: hallucinations 

* try to categorize them like in [2] which makes sense 
* go over some prominent examples (just a couple of examples, there are to many models out there to cover them all)



## Families of VLMs 

- categorized based on the training paradigm:
    - contrastive: leverage pairs of positive and negative examples 
    - masking: leverages reconstruction of masekd image patches given some unmasked text  
    - pretrained backbones: LLM + pretrained image encoder, learn mapping between those two (less computationally expensive) 
    - generative: generate captions and images (expensive to train)
- paradigms are not mutually exclusive because many approaches rely on a mix of those training strategies 

{{< figure align=center alt="Families of Vision Language Models" src="/imgs/vlms/families_of_vlms.png" width=100% caption="Figure 1. Families of VLMs [2]">}}

<p align="justify">
</p>

### Contrastive-based Methods 

<p align="justify">
</p>

#### CLIP 

- CLIP [3] (2021)
    - pretraining task: predict which caption goes with which image 
    - trained on 400 million image-text-pairs from the web, the dataset was called WIT for WebImageText 

{{< figure align=center alt="CLIP approach overview" src="/imgs/vlms/clip.png" width=100% caption="Figure 2. CLIP approach overview [3]">}}
    - leveraging captions from the internet instead of classic machine learning labels like in classification tasks makes it easier to collect a big dataset and it learns a connection between image and text which enables a flexible zero-shot transfer 
    - training proxy task: predict which text as a whole is paired with which image 
    - switiching from a predictive objective to a contrastive objective create a 4x efficiency improvement
    - training procedure: given a batch of N image-text-pairs, CLIP is trained to predict which of the NxN possible image-text-pairings across a batch actually occured. to accomplish this, clip learns a multimodal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and the text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the incorrect pairings
    - a symmetric cross entropy loss over the similarity scores is used for optimization
    - the following pseudocode describes this procedure: 

{{< figure align=center alt="CLIP training pseudocode" src="/imgs/vlms/clip_code.png" width=60% caption="Figure 3. CLIP training pseudocode [3]">}}

- as the image encoder, the authors trained 5 ResNet and 3 ViT versions and found the ViT-L/14@336px version to perform the best 
- the text encoder is a 63M parameter Transformer (12 layers) with a vocab size of ~49K 
- remember those days when OpenAI was open and cool and giving us all this information, they used to be cool, believe me 



### VLMs with Masking Objectives 


### Generative-based VLMs


### VLMs from Pretrained Backbones 

* InternVL1.5 & 2
* Idefics 1 & 2 & 3


## Training 



## Evaluation 



## Leaderboard 

* let's take a look at the current Huggingface VLM Leaderboard



<p align="justify">
</p>

## References 

<a name="references"></a>

[[1]](https://huggingface.co/blog/vlms) M. Noyan & E. Beeching "Vision Language Models Explained" (2024).

[[2]](https://arxiv.org/pdf/2405.17247) Bordes et al. "An Introduction to Vision-Language Modeling" (2024) 

[[3]](https://arxiv.org/pdf/2103.00020) Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" (2021)