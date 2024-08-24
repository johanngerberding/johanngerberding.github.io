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

* current problems of VLMs: 
    - understanding spatial relationships
    - counting stuff (without complicated engineering overhead that relies on additional data annotation)
    - lack understanding of attributes and ordering
    - ignorance of parts of the input prompt (a lot of prompt engineering is needed to produce the results you want)
    - classic: hallucinations 






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

<p align="justify">
</p>

## References 

<a name="references"></a>

[[1]](https://huggingface.co/blog/vlms) M. Noyan & E. Beeching "Vision Language Models Explained" (2024).

[[2]](https://arxiv.org/pdf/2405.17247) Bordes et al. "An Introduction to Vision-Language Modeling" (2024) 
