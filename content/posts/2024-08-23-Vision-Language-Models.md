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

<p align="justify">
Vision Language Models (VLM) are multimodal models that can learn from text and images and generate text as an output (some are able to generate images too). Typical use cases for this range from image chatting, image recognition, visual question answering to image captioning, document understanding and more. In addition to that some models are also able to perform object detection, segmentation or reasoning about relative positions of objects (which is kinda fire).
</p>

<p align="justify">
The focus of this post lies on open source models just because we have information about their architecture, datasets and training processes. VLMs currently are far from being perfect, there exist a lot of open questions, challenges when building them and problems that have to be addressed e.g.:
</p>
<ul>
<li>bad understanding of spatial relationships</li>
<li>bad at counting (without complicated engineering overhead that relies on additional data annotation or other hacks)</li>
<li>lack understanding of attributes and ordering</li>
<li>ignorance of parts of the input prompt (need for a lot of prompt engineering to produce the results you want)</li>
<li>hallucinations (like in LLMs)</li>
</ul> 
    
<p align="justify">
In the following I will categorize them by their training paradigm like in [2] and will go over some prominent examples. There exist way to many models to cover them all so if this overview here isn't enough and you want more information check out <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending">huggingface models</a>, this cool <a href="https://huggingface.co/collections/merve/vision-language-models-papers-66264531f7152ac0ec80ceca">paper collection</a> or <a href="https://paperswithcode.com/">paperswithcode</a>. 
</p>


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

{{< figure align=center alt="Families of Vision Language Models" src="/imgs/vlms/families_of_vlms.png" width=90% caption="Figure 1. Families of VLMs [2]">}}

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
</p>
<p align="justify">
The image encoder is a ViT-B/16 model with a fixed image size. It outputs is a list of hidden state vectors $\{h_{I}\}$, each corresponding to an image patch, and a classification token $h_{CLS,I}$. The text encoder has the same architecture as the vision part and outputs a hidden state vector $\{h_{T}\}$ and a text classification token $h_{CLS,T}$. The multimodal encoder also is a transformer model that fuses image and text hidden states. Over each of the hidden state vectors generated by the image and text encoder two learned linear projections are applied and an additional $[CLS_M]$ token is added before feeding this into the multimodal encoder. Like the text and image encoders, the multimodal encoder also outputs a list of hidden state vectors $\{h_{M}\}$ and a vector $h_{CLS,M}$ for the classification token. 
</p>

{{< figure align=center alt="Overview of the FLAVA model architecture" src="/imgs/vlms/flava_overview.png" width=100% caption="Figure 6. Overview of the FLAVA model architecture [5]">}}

<p align="justify">
The training process consists of a joint pretraining on both unimodal and multimodal data. First the authors pretrained the text encoder, then the image encoder and then they used both pretrained encoders to train the multimodal encoder. 
For the unimodal pretraining they used established loss components. For the image encoder they used a masked image modeling (MIM) objective like in BEiT. First they tokenize the image patches with a dVAE tokenizer, then they used a classifier on top of the image encoder outputs to predict the missing tokens. Masked language modeling (MLM) was used for the text encoder as an objective with 15% of text tokens masked. For the multimodal pretraining three loss components were used:
</p>

<ul>
<li><b>Global contrastive</b> (GC) loss, like in CLIP</li>
<li><b>Masked Multimodal Modeling</b> (MMM) loss, masking of both the image patches and text tokens</li>
<li><b>Image Text Matching</b> (ITM) loss, by applying a classifier on top of the multimodal encoder to decide if the input image and text match to each other</li>
</ul>

<p align="justify">
One of the cool things of this paper is, that the authors only used public unimodal and multimodal datasets for training, 70 million image-text pairs in total (but the avg. text length was just 12 words). The validated FLAVA on 35 tasks across vision, NLP and multimodal domains and performed better or competitive on all of those tasks with the state of the art models at that time which were mostly trained on much larger and probably cleaner datasets. 
</p>

#### MaskVLM

<p align="justify">
Instead of developing masked language modeling (MLM) and masked image modeling (MIM) independently, the authors propose to build joint masked vision and language modeling, where the masked signal of one modality is reconstructed with the help from another modality. The masked signal reconstruction of one modality conditioned on another modality can also implicitly learn cross-modal alignment between language tokens and image patches. This works especially well in scenarios with limited data. Figure 7 illustrates the difference between this new paradigm and the classic MIM and MLM based approaches.
</p>

{{< figure align=center alt="Left MIM and MLM and right the MaskVLM idea" src="/imgs/vlms/maskvlm_idea.png" width=85% caption="Figure 7. Left: MIM & MLM; Right: Masked Vision Language Modeling [6]">}}

<p align="justify">
There are two main types of pre-training objectives in this model. The first is masked vision and language modeling. Here, transformer-based models are used as image and text encoders to extract features from both modalities. These features are then processed by image and text cross-modality encoders, which consist of three cross-attention blocks. These blocks allow the text and image features to interact, enhancing the representation of each by leveraging information from the other. The second objective is multimodal alignment, which includes image-text contrastive learning (ITC) and image-text matching (ITM). These methods align the representations of images and text to ensure they correspond accurately. For more detailed information, you can refer to the original paper.
</p>

{{< figure align=center alt="MaskVLM model architecture" src="/imgs/vlms/maskvlm_architecture.png" width=90% caption="Figure 8. Overview of the MaskVLM model architecture [6]">}}

<p align="justify">
MaskVLM is very data efficient, it especially shines in limited data scenarios where only ∼40% of data used by the state-of-the-art models is sufficient to match their performance.
</p>

### Generative-based VLMs

<p align="justify">
In contrast to the paradigms above, that mostly operate on latent representations we will now look at generative VLM that are trained to generate text and images. We are looking at two methods in more detail, <b>CoCa</b> which learns to generate text and <b>Chameleon</b> which is a multimodal generative model that can generate text and images. Before we delve deeper I will list some of the advantages of generative classifiers and why this training paradigm can be a good idea: 
</p>

<ul>
<li>more effective robustness which means better out-of-distribution performance</li>
<li>better on compositional reasoning tasks than discriminative methods like CLIP</li> 
<li>more shape bias and better alignment with human judgement</li>
<li>can be jointly adapted with discriminative models at test time using only unlabeled test samples which improves classification, segmentation and depth prediction performance</li>
</ul>

#### CoCa 

- <b>Co</b>ntrastive <b>Ca</b>ptioner (CoCa) is an image-text encoder-decoder foundation model pretrained jointly with a contrastive and a captioning loss, which makes it a combination of contrastive approaches like CLIP and generative methods. 
- decoder is decoupled into two parts, a unimodal decoder and a multimodal decoder
- omit cross-attention in unimodal decoder layers to encode text-only representations, and cascade multimodal decoder layers cross-attending to image encoder outputs to learn multimodal image-text representation


<p align="justify">
</p>

{{< figure align=center alt="CoCa model architecture, training objectives and pseudocode" src="/imgs/vlms/coca_architecture.png" width=100% caption="Figure 9. Overview of the CoCa model architecture and training objectives [10]">}}


<p align="justify">
- quick transfer to downstream tasks with zero-shot transfer or minimal task adaptation
</p>

#### CM3leon 

<p align="justify">
- retrieval augmented, token based, decoder-only multimodal model capable of generating text and images
- uses the CM3 model architecture as basis and benefits a lot from scaling and instruction tuning
- CM3 uses a VQGAN to turn images into 1024 tokens 
- trained with a recipe adapted from text-only language models, including a large scale retrieval-augmented pretraining stage and a second multi-task supervised finetuning stage
</p>

{{< figure align=center alt="RA-CM3 model architecture" src="/imgs/vlms/ra-cm3_architecture.png" width=100% caption="Figure 10. Overview of the RA-CM3 model architecture and training pipeline [12]">}}

<p align="justify">
</p>

### VLMs from Pretrained Backbones 

<p align="justify">
- costly to train from scratch because you need hundreds or thousands of GPUs while having to use millions of image-text pairs 
- to avoid these high costs there is a lot of research in the area of leveraging existing unimodal models 
- just learn to map between the text and image modalities which requires a low amount of compute resources
- in this section we are looking at some of the best open source vision language models out there: the Idefics series, InternVL1.5 and 2, Qwen2-VL
</p>

#### Idefics 

Idefics1: 
* introduction of the OBELICS dataset which consists of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images and 115 billion text tokens
* one of the advantages of OBELICS is the amount and detail of text per image, which is much bigger than in other image-text datasets
* Figure X down below shows the whole dataset creation process  

{{< figure align=center alt="Generation process of the OBELICS dataset" src="/imgs/vlms/obelics.png" width=90% caption="Figure 9. Overview of the OBELICS generation process [7]">}}

* with that dataset the authors created two vision language models called "Idefics", a 9 and a 80 billion parameter model  
* it is based on the Flamingo architecture, comprised of two frozen unimodal backbones, Llama for the language encoder and OpenCLIP for the vision part
* learnable cross-attention Transformer blocks and Perceiver blocks are added to connect both unimodal encoders
* best results on a combination of the LAION and the OBELICS datasets


Idefics2: 
* investigate which choices matter when building VLMs
* focus on two design choices 
    * model architecture, especially strategy of how to fuse image and text information 
    * training procedure
* main findings:
    - progress in vision-language models is largely driven by the progress of pretrained unimodal backbones (LLM is more important than Vision Encoder, but there are no really good vision encoders out there and I think InternVL will show that this finding is not true) 
    - fully autoregressive architecture outperforms cross-attention, but it requires modifications to the optimization procedure to ensure a stable training 
    - reducing the number of visual tokens with learned pooling significantly improves compute efficiency while also improving performance on downstream tasks (efficiency yes but performance is another thing which is hard to prove without a very good vision encoder)
    - splitting images into subimages allow trading compute efficiency for more performance during inference, especially noticeable for tasks that require text reading capabilities

* based on the findings described above the authors created Idefics2, a 8B parameter VLM that achieves state-of-the-art performance within its size category 

{{< figure align=center alt="Idefics2 model architecture" src="/imgs/vlms/idefics2.png" width=90% caption="Figure 10. Idefics2 model architecture [8]">}}

* vision encoder = SigLIP-SO400M, LLM = Mistral-7B-v0.1
* datasets for training: OBELICS, LAION COCO, PMD, OCR-IDL, PDFA 

Idefics3: 
* test
<p align="justify">
</p>

#### InternVL

<p align="justify">
</p>

#### Qwen2-VL

<p align="justify">
</p>

#### MiniCPM-V 

2.5 & 2.6
<p align="justify">
- remaining challenges prevent VLMs from being used in real world applications, the most significant one is the high cost of running those big models 
- most VLMs have to be deployed on high-performance cloud servers, which greatly limits their application scope (mobile, offline, privacy-protective)
- MiniCPM is a model family that tries to change that
- model have strong performance on general benchmarks and especially OCR capabilities, has multilingual support for more than 30 languages and you can run these models on mobile phones
</p>

{{< figure align=center alt="MiniCPM-V-2.5 model architecture" src="/imgs/vlms/minicpm.png" width=100% caption="Figure 11. MiniCPM-V-2.5 model architecture []">}}

<p align="justify">
- three key modules: visual encoder, compression layer, LLM 
- input image is encoded by a SigLIP SoViT-400m/14, utilizing the adaptive visual encoding approach proposed by LLaVA-UHD
- the compression layer has a perceiver resampler structure with one layer cross-attention
- the compressed visual tokens along with the text input are fed into the LLM for conditional text generation
- image partitioning -> slice encoding (and resizing so that the slice size matches the ViT input size) -> token compression (1024 tokens per image slice, compressed with cross attention layer to 64/96 tokens)
- 3 phase training process: pretraining, supervised fine-tuning, RLAIF-V
- RLAIF-V is a framework for improving trustworthiness and reduce hallucinations 
</p>


## Training 

<p align="justify">
- scale is very important to model performance but a lot of companies and academic labs don't have the resources to aquire the amount of compute needed to train such models
- recently researchers found that a data curation pipeline makes it possible to beat the scaling law
</p>

{{< figure align=center alt="Important training decisions to make" src="/imgs/vlms/training.png" width=100% caption="Figure X. Important decisions to make when training VLMs [2]">}}

<p align="justify">
- in the following we will explore the importance of data and recipes that are used to create high quality datasets for VLM training
- in addition we will go into grounding, alignment with human preferences and some recipes about choosing the right model for your use case
</p>

### Data

<p align="justify">
The quality and organization of data play a critical role in the successful training of Vision-Language Models (VLMs). Due to the complex nature of these models, which need to understand both visual and textual inputs, effective data management through pruning, augmentation, and quality control is essential to achieving optimal performance. Data pruning is a crucial step in refining the datasets used to train VLMs. Pruning techniques fall into three main categories:
</p>
<ul>
<li><b>Heuristic-based pruning</b> involves the use of rules to eliminate low-quality image-text pairs. This can be done through unimodal filtering, where low-quality captions (e.g., lacking complexity or containing non-English text) or inappropriate images (based on size or aspect ratio) are removed. Multimodal filtering further refines data by using image classifiers to detect when objects in the image do not align with the associated text, or by filtering out examples where the majority of text in the caption appears directly within the image itself.</li>
<li><b>Bootstrapping methods</b> leverage pre-trained VLMs to rank image-text pairs based on their multimodal alignment, using metrics such as CLIPScore. This approach helps prioritize data that better represents meaningful relationships between the visual and textual components, further refining the dataset for training.</li>
<li><b>Diversity and balance techniques</b> aim to create datasets that not only contain high-quality pairs but also represent a wide variety of content. Ensuring diversity in objects, actions, and attributes across different modalities can prevent bias and improve the model’s generalization capability.</li>
</ul>

<p align="justify">
In addition to pruning, the generation of synthetic data has become an effective strategy for enhancing VLM performance. For example, LLMs can generate captions, which can then be used by text-to-image models to create corresponding images. This helps to supplement the training data, especially in areas where real-world examples may be limited or lacking. Data augmentation techniques, such as adding variations to existing image-text pairs (e.g., rotating or cropping images, rewording captions), further contribute to improving model robustness by exposing the VLM to a wider range of inputs.
</p>
<p align="justify">
Data quality is paramount when training VLMs or any other AI model, but assessing the quality of multimodal and interleaved data remains a challenge. The lack of a standardized method for evaluating such data complicates the task, as quality often depends on the subjective judgment of what constitutes a "good" image-text pair. This is an active area of research, as models continue to evolve and demand increasingly refined datasets for training. Despite the challenges, curated datasets such as OBELICS, which contain interleaved data from multiple modalities, have shown promising results. These datasets are carefully constructed to ensure a rich variety of content, making them valuable resources for VLM training.
</p>
<p align="justify">
In summary, data plays a foundational role in training vision-language models, and techniques such as data pruning, augmentation, and synthetic data generation are essential for improving VLM performance. However, assessing data quality remains an open challenge that continues to evolve alongside advancements in multimodal learning.
</p>

### Grounding 

<p align="justify">
- grounding describes the ability to correctly map text with visual clues 
</p>

### Alignment

<p align="justify">
</p>

## Evaluation 

<p align="justify">
</p>

### Benchmarks

<p align="justify">
</p>


### HuggingFace VLM Leaderboard 

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

[[7]](https://arxiv.org/pdf/2306.16527) Laurencon et al. "OBELICS: An Open Web-Scaled Filtered Dataset of Interleaved Image-Text Documents" (2023) 

[[8]](https://arxiv.org/pdf/2405.02246) Laurencon et al. "What matters when building vision-language models?" (2024)

[[9]](https://arxiv.org/pdf/2408.12637) Laurencon et al. "Building and better understanding vision-language models: insights and future directions" (2024)

[[10]](https://arxiv.org/pdf/2205.01917) Yu et al. "CoCa: Contrastive Captioners are Image-Text Foundation Models" (2022)

[[11]](https://arxiv.org/pdf/2309.02591) Yu et al. "Scaling Autoregressive Multi-Modal Models: Pre-Training and Instruction Tuning" (2023)

[[12]](https://arxiv.org/pdf/2211.12561) Yasunaga et al. "Retrieval-Augmented Multimodal Language Modeling" (2023)

[[13]](https://arxiv.org/pdf/2201.07520) Aghajanyan et al. "CM3: A Causal Masked Multimodal Model of The Internet" (2022)