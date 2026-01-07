---
title: Paper Review - DeepSeek-OCR  
date: 2026-01-07 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: Let's take a closer look at the DeepSeek-OCR model and what's so special about it. 
include_toc: true
showToc: true
math: true
draft: true
tags: ["vlm", "transformer", "encoder-decoder", "ocr", "compression"]
---

## Introduction

<p align="justify">
- late to the party with this one 
- struggle to write at all because of the length and research effort of previous posts 
- change the mode to shorter form paper reviews
- not really possible to keep up with papers etc. so lets focus on certain things an go deep
- lets start with looking into the DeepSeek-OCR model, which is an interesting one in terms of efficiency and architectural details
</p>


## Problem & Innovation  

<p align="justify">
- computational challenges when processing long text content due to quadratic scaling with sequence length 
- proposed solution: image containing text can represent rich information with less tokens than the text itself -> optical compression through vision tokens may be better and more efficient than text 
- OCR acts as a testbed for that idea, because of the straightforward quantitative evaluation possibility 
- three main contributions:
    - show that vision-text token compression is more efficient (and even qualitatively better, think about diagrams etc.) 
    - new Vision encoder architecture called DeepEncoder, which maintains low activation memory and minimal vision tokens even with high resolution inputs 
    - DeepSeek-OCR, consisting of a DeepEncoder and the DeepSeek3B-MoE, achieving state-of-the-art performance on OmniDocBench while using the fewest vision tokens 
- 7-20x token reduction potential, details on this later
</p>


## How does it work   

<p align="justify"></p>

### Architecture 

<p align="justify">
- unified end-to-end VLM architecture consisting of an encoder and a decoder
- encoder is called DeepEncoder that extracts image features, tokenizing and compressing visual representations 
- decoder is used for generating the required result based on image tokens and prompts 
</p>

{{< figure align=center alt="DeepSeek-OCR model architecture" src="/imgs/deepseek_ocr/deepseek-ocr-architecture.png" width=100% caption="Figure 1. DeepSeek-OCR model architecture">}}

<p align="justify">
- DeepEncoder is 380M parameters, composed of an 80M SAM-base and a 300M CLIP-large connected in series 
- this encoder is capable of processing high resolutions, has low activations at the same time, produces just a few vision tokens, supports a mutli-resolution inputs and has a moderate parameter count 
</p>

<p align="justify">
- decoder adopts the DeepSeek-3B-MoE architecture with 570M activated parameters 
- during inference, the model activates 6 out of 64 routed experts and 2 shared experts 
</p>

### Training 


<p align="justify">
- they define two types of OCR data used for training the model: OCR 1.0 data, which is the traditional OCR task data and OCR 2.0 data, which includes parsing tasks for complex artificial images like common charts, chemical formulas and plane geometry parsing data 
- 2-stage training pipeline, first training the DeepEncoder independently and second training DeepSeek-OCR end-to-end 
</p>

## Results 

<p align="justify">
</p>

## Limitations 

<p align="justify">
</p>


## Conclusion 

<p align="justify">
</p>


## References 

[[1]](https://arxiv.org/pdf/2510.18234) Haoran Wei, Yaofeng Sun, Yukun Li "DeepSeek-OCR: Contexts Optical Compression" (2025). 

[[2]](https://arxiv.org/pdf/2405.04434) DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024). 

[[3]](https://arxiv.org/pdf/2412.19437) DeepSeek-AI "DeepSeek-V3 Technical Report" (2025). 
