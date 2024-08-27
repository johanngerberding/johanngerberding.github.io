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


## Introduction  

What is a VLM? 
* multimodal models that can learn from images and text 
* generative models
    - input = image, text 
    - output = text
* use cases: image chatting, image recognition, visual question answering, document understanding, image captioning and further more
* some of them are able to perform object detection, segmentation or reasoning about relative positions of objects (which is kinda fire)


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

<p align="justify">
One way to categorize VLMs is based on the training paradigm like in [2]:
</p>
<ul>
<li><b>contrastive</b>: Leverage pairs of positive and negative examples.</li>
<li><b>masking</b>: Leverage reconstruction of masked image patches given some unmasked text.</li>
<li><b>pretrained backbones</b>: Combine a pretrained LLM with a pretrained image encoder and then learn a mapping between those two.</li>
<li><b>generative</b>: Generate captions and images.</li>
</ul>

{{< figure align=center alt="Families of Vision Language Models" src="/imgs/vlms/families_of_vlms.png" width=100% caption="Figure 1. Families of VLMs [2]">}}

<p align="justify">
The paradigms are not mutually exclusive and many approaches we explore in this post rely on a mix of those training strategies. In the following we will describe some approaches for each paradigm.
</p>

### Contrastive-based Methods 

<p align="justify">
In this section I am presenting two contrastive-based VLMs, the very popular CLIP model from OpenAI and one of the successor models from Meta called Llip.
</p>

#### CLIP 

<p align="justify">
CLIP (<b>C</b>ontrastive <b>L</b>anguage <b>I</b>mage <b>P</b>re-training) was one of those models created by OpenAI which were really open (you know back in the days OpenAI was really open and cool). The pre-training task here was not to predict the exact caption for each image but to predict which whole caption to pair with a certain image. This switch from a predictive objective (so the classic ML approach with labels) to a contrastive one lead to a 4x efficiency improvement. To train this model the authors leveraged captions from the internet and collected a huge dataset of 400 million image-text-pairs which they called WIT for WebImageText. Another advantage of prediction captions instead of e.g. classes was a better and more flexible zero-shot transfer capability of the model. The figure down below gives a good overview of the approach.
</p>

{{< figure align=center alt="CLIP approach overview" src="/imgs/vlms/clip.png" width=100% caption="Figure 2. CLIP approach overview [3]">}}

<p align="justify">
The CLIP model is trained using a batch of $N$ image-text pairs. The training objective is to predict which of the $N×N$ possible image-text pairings within the batch actually occurred. To achieve this, CLIP learns a multimodal embedding space by jointly training an image encoder and a text encoder. The goal is to maximize the cosine similarity between the image and text embeddings of the $N$ correct (real) pairs in the batch while minimizing the cosine similarity for the incorrect pairings. For optimization, a symmetric cross-entropy loss, also known as InfoNCE loss, is applied to the similarity scores. The following pseudocode outlines this procedure.
</p>

{{< figure align=center alt="CLIP training pseudocode" src="/imgs/vlms/clip_code.png" width=60% caption="Figure 3. CLIP training pseudocode [3]">}}

<p align="justify">
As the image encoder, the authors trained 5 ResNet and 3 ViT versions and found the ViT-L/14@336px version to perform the best. The text encoder is a 63M parameter Transformer (12 layers) with a vocab size of ~49K. The model showed strong zero-shot performance on ImageNet classification task (same as the original ResNet50). Some of the limitations of CLIP were e.g. that the zero-shot performance on more finegrained vision tasks was quite bad (like differenciating between different car models), on some tasks it was random (like counting objects) and it was not as good as you would expect on simple out of distribution tasks like MNIST (just 88% accuracy).  
</p>

#### Llip 


<p align="justify">
One of the problems with CLIP was, that there are a thousand ways to caption an image, based on the fact that the caption could describe only specific regions of an image or specific objects. To better model the visual richness of an image, a training objective of a vision language model should aim to capture all the possible text descriptions. This is what the authors of Llip, <b>L</b>atent <b>L</b>anguage <b>I</b>mage <b>P</b>retraining, try to do. To enable the prediction of different representations from a fixed image, they implemented the image-to-text representation function as a one-to-many mapping. This is achieved by augmenting the visual encoder with a latent variable that captures context information. The contextual latent is inferred from the caption and used to modulate the representation. The visual encoder is implemented as a Vision Transformer that outputs $K$ learnable mixture tokens in addition to the visual tokens. These mixture tokens should capture different visual aspects of an image. Figure 4 down below shows this simple modification of CLIP.
</p>

{{< figure align=center alt="CLIP vs. Llip" src="/imgs/vlms/clip_vs_llip.png" width=80% caption="Figure 4. CLIP vs. Llip [4]">}}

<p align="justify">
The authors added a cross-attention mechanism to infer the mixture token weights as a function of the text caption. The weighted mixture defines the contextual representation that is contrasted with text representations. This leads to significant improvement of the visual representation quality as well as a more rich visual representation.
</p>

{{< figure align=center alt="Llip Cross Attention mechanism" src="/imgs/vlms/llip_cross_attention.png" width=90% caption="Figure 5. Llip cross-attention mechanism [4]">}}

<p align="justify">
On zero-shot transfer classification, Llip consistently outperforms CLIP pretraining for architecture of similar size on a large set of benchmarks. Especially on zero-shot image-text and text-image retrieval, Llip consistently outperforms CLIP pretraining on COCO by 6.0% image-to-text retrieval.
</p>

### VLMs with Masking Objectives 

<p align="justify">
Masking is a commonly used technique in deep learning research. It can be viewed as a specific form of denoising autoencoder in which the noise has a spatial structure. In 2019 the authors of the BERT paper used Masked-Language-Modeling (MLM) to predict missing text tokens in a sentence. More recently the same concept (Masked-Image-Modeling) was used in the vision space to learn strong visual representations like in I-JEPA. In the following we are going through two approaches that combined those techniques to train a VLM, FLAVA [5] and MaskVLM [6].
</p>

#### FLAVA 

<p align="justify">
Contrastive methods like CLIP aren't easy usable on multimodal problems that require dealing with both modalities at the same time. Many of the more recent models that rely on early fusion and shared self-attention across modalities often perform very bad on vision-only or language-only tasks. The goal of the authors was to create a single "foundation" model that is good at vision tasks, language tasks and cross- and multi-modal tasks. FLAVA consists of three models, an image encoder, a text encoder and a multimodal encoder that takes as input the encoded image and text and integrates their represenations for multimodal reasoning.

- image encoder: ViT-B/16 with a fixed image size, outputs is a list of hidden state vectors $\{h_{I}\}$, each corresponding to an image patch + classification token $h_{CLS,I}$    
- text encoder: tokenize + embed + transformer -> hidden state vector $\{h_{T}\}$ + text classification token $h_{CLS,T}$, same architecture as vision part ViT-B/16
- multimodal encoder: transformer to fuse image and text hidden states; with learned linear projections over each hidden state vector     
</p>

{{< figure align=center alt="Overview of the FLAVA model architecture" src="/imgs/vlms/flava_overview.png" width=100% caption="Figure 6. Overview of the FLAVA model architecture [5]">}}

<p align="justify">

- training process: joint pretraining on both unimodal and multimodal data while encompassing cross-modal alignment objectives and multimodal fusion objectives
- training data is openly available

- validation of FLAVA on 35 tasks across vision, NLP and multimodal domains
</p>

#### MaskVLM

<p align="justify">
</p>

### Generative-based VLMs

<p align="justify">
</p>

### VLMs from Pretrained Backbones 

<p align="justify">
</p>

#### Idefics 

<p align="justify">
</p>

#### InternVL

<p align="justify">
</p>

## Training 

<p align="justify">
</p>

## Evaluation 

<p align="justify">
</p>

## Leaderboard 

* let's take a look at the current Huggingface VLM Leaderboard



<p align="justify">
</p>

## References 

<a name="references"></a>

[[1]](https://huggingface.co/blog/vlms) M. Noyan & E. Beeching "Vision Language Models Explained" (2024).

[[2]](https://arxiv.org/pdf/2405.17247) Bordes et al. "An Introduction to Vision-Language Modeling" (2024) 

[[3]](https://arxiv.org/pdf/2103.00020) Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" (2021)

[[4]](https://arxiv.org/pdf/2405.00740) Lavoie et al. "Modeling Caption Diversity in Contrastive Vision-Language Pretraining" (2024)

[[5]](https://arxiv.org/pdf/2112.04482) Singh et al. "FLAVA: A Foundational Language And Vision Alignment Model" (2021)

[[6]](https://arxiv.org/pdf/2208.02131) Kwon et al. "Masked Vision and Language Modeling for Multi-Modal Represenation Learning" (2023)