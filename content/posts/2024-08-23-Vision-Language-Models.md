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

<p align="justify">
The <b>Contrastive Captioner (CoCa)</b> is an advanced image-text encoder-decoder foundation model that integrates both contrastive and captioning techniques. Pretrained with a combination of contrastive loss and captioning loss, CoCa combines contrastive methods like CLIP with generative approaches. In CoCa, the decoder is split into two components: a <b>unimodal decoder</b> and a <b>multimodal decoder</b>. The unimodal decoder is responsible for encoding text-only representations and omits cross-attention, while the multimodal decoder employs cross-attention to interact with image encoder outputs, learning multimodal image-text representations. The model is trained with a contrastive loss between unimodal image and text embeddings, alongside a captioning loss applied to the multimodal decoder's outputs, which autoregressively predict text tokens. By sharing the computational graph, CoCa efficiently computes both training objectives with minimal overhead.
</p>

{{< figure align=center alt="CoCa model architecture, training objectives and pseudocode" src="/imgs/vlms/coca_architecture.png" width=100% caption="Figure 9. Overview of the CoCa model architecture and training objectives [10]">}}


<p align="justify">
CoCa is pretrained from scratch by treating annotated image labels as text. This pretraining process utilizes two major datasets: ALIGN, which contains approximately 1.8 billion images paired with alt-text, and JFT-3B, an internal dataset with over 29.5k classes. In the case of JFT-3B, the labels are treated as alt-text for training purposes. This pretraining setup enables CoCa to transfer quickly to downstream tasks, either through zero-shot transfer or with minimal task-specific adaptation, making it highly versatile for a range of applications.
</p>

#### CM3leon 

<p align="justify">
CM3leon is a retrieval-augmented, token-based, decoder-only multimodal system designed for generating both text and images. It is built upon the CM3 model architecture, which has shown significant benefits from scaling and instruction tuning. In CM3, images are transformed into 1024 tokens using a VQGAN (Vector Quantized Generative Adversarial Network). The training process follows a recipe adapted from text-only language models. This includes a large-scale retrieval-augmented pretraining phase, followed by a second stage of multi-task supervised finetuning to enhance the model’s performance across different tasks.
</p>

{{< figure align=center alt="RA-CM3 model architecture" src="/imgs/vlms/ra-cm3_architecture.png" width=100% caption="Figure 10. Overview of the RA-CM3 model architecture and training pipeline [12]">}}

<p align="justify">
The training objective for the model is based on an infill approach, where specific spans are masked and relocated to the end of the sequence, followed by the use of standard next-token prediction loss. Despite using only a fraction of the training data and compute, the model achieves zero-shot results that are on par with state-of-the-art models, such as on the MS-COCO benchmark. The authors further fine-tuned the model for instructable image generation and conditional text generation, yielding similar results as in the zero-shot settings. Notably, the model can compete with state-of-the-art systems like Flamingo and OpenFlamingo, even though those models were exposed to significantly more tokens (3B vs. 100B or 40B).
</p>

### VLMs from Pretrained Backbones 

<p align="justify">
Training models from scratch is prohibitively expensive, requiring hundreds or even thousands of GPUs and millions of image-text pairs. To mitigate these high costs, a growing area of research focuses on leveraging existing unimodal models rather than building new systems from the ground up. This approach involves learning to map between the text and image modalities, which significantly reduces the amount of compute resources required. In this section, we will explore some of the best open-source vision-language models available, including the Idefics series, InternVL1.5 and 2, Qwen2-VL and MiniCPM-2.6.
</p>

#### Idefics 

<p align="justify">
The Idefics model family currently includes three versions that build on one another, each advancing the capabilities of the previous model.
</p>

<p align="justify">
<b>Idefics1</b> introduces the OBELICS dataset, a significant milestone in this model family. The dataset is composed of interleaved image-text documents, derived from 141 million web pages extracted from Common Crawl. It contains a total of 353 million associated images and 115 billion text tokens. One of the standout features of the OBELICS dataset is the extensive detail and quantity of text per image, which surpasses other image-text datasets in size and richness. Figure X below illustrates the complete dataset creation process.
</p>

{{< figure align=center alt="Generation process of the OBELICS dataset" src="/imgs/vlms/obelics.png" width=90% caption="Figure 9. Overview of the OBELICS generation process [7]">}}

<p align="justify">
Using the OBELICS dataset, the authors developed two vision-language models under the name "Idefics," featuring versions with 9 billion and 80 billion parameters. These models are based on the Flamingo architecture, which integrates two frozen unimodal backbones: Llama, serving as the language encoder, and OpenCLIP, handling the vision component. To bridge these two encoders, the model incorporates learnable cross-attention Transformer blocks and Perceiver blocks, enabling effective communication between the unimodal systems. The Idefics models demonstrate top performance when evaluated on a combination of the LAION and OBELICS datasets.
</p>

<p align="justify">
<b>Idefics2</b> focuses on exploring key factors that influence the development of VLMs, specifically investigating two critical design choices: the model architecture — particularly how image and text information are fused — and the training procedure. The main findings from this research are as follows:
</p>

<ul>
<li>Progress in vision-language models is largely driven by advances in pretrained unimodal backbones. Notably, the language model (LLM) plays a more crucial role than the vision encoder, although there is currently a lack of high-quality vision encoders. The authors suggest that models like InternVL could challenge this conclusion.</li>
<li>Fully autoregressive architectures outperform cross-attention models, but they require adjustments to the optimization process to maintain training stability.</li>
<li>Reducing the number of visual tokens through learned pooling significantly enhances computational efficiency and improves performance on downstream tasks. However, while efficiency gains are clear, performance improvements are difficult to validate without a superior vision encoder.</li>
<li>Splitting images into subimages allows for a trade-off between computational efficiency and performance during inference, particularly for tasks that involve text reading.</li>
</ul>

<p align="justify">
Based on these insights, the authors developed Idefics2, an 8-billion parameter vision-language model that achieves state-of-the-art performance within its size category.
</p>

{{< figure align=center alt="Idefics2 model architecture" src="/imgs/vlms/idefics2.png" width=90% caption="Figure 10. Idefics2 model architecture [8]">}}

<p align="justify">
The vision encoder used in Idefics2 is SigLIP-SO400M, while the LLM is Mistral-7B-v0.1. The model was trained using a variety of datasets, including OBELICS, LAION COCO, PMD, OCR-IDL, and PDFA.
</p>

<p align="justify">
<b>Idefics3</b> marks a significant improvement over its predecessor, Idefics2, particularly in tasks related to document understanding. This model was exclusively trained on open datasets, showcasing its accessibility and transparency. One of the key advancements in Idefics3 is the introduction of the Docmatix dataset, which includes 2.4 million images and 9.5 million question-answer pairs derived from 1.3 million PDF documents. For more details on how this dataset was created, the authors recommend consulting the original paper.
</p>

<p align="justify">
Idefics3 is built on Llama 3.1 Instruct for the language model and retains SigLIP-SO400M as the vision encoder. A notable architectural change from Idefics2 is the replacement of the perceiver resampler with a simpler pixel shuffle strategy. This technique acts as a pooling method that reduces the number of image hidden states by a factor of four, encoding images of up to 364x364 pixels into 169 visual tokens.
</p>

<p align="justify">
During both training and inference, the model utilizes an image-splitting strategy, where the original image is divided into a matrix of tiles, each measuring 364x364 pixels, and the downscaled image is appended at the end. The training process consists of three pre-training stages, followed by supervised fine-tuning.
</p>

{{< figure align=center alt="Idefics3 training stages overview" src="/imgs/vlms/idefics3_training.png" width=90% caption="Figure 11. Overview of Idefics3 training stages [9]">}}

<p align="justify">
In the first pretraining stage of Idefics3, the model’s backbones remain frozen to maintain their performance while the newly initialized parameters are learned. During this phase, the maximum image resolution is gradually increased from 364x364 to 1820x1820 pixels. Starting from the second stage, the backbones are trained using DoRA (a variant of LoRA, explained in the Training section) and with larger images. The final pretraining stage focuses on using large synthetic datasets for further training.
During the supervised fine-tuning stage, NEFTune noise is applied to the inputs, and the loss is calculated only on the answer tokens to refine performance.
</p>

<p align="justify">
The authors evaluated Idefics3 on several common benchmarks, such as MMMU, MathVista, MMStar, DocVQA, and TextVQA. The most notable improvements over Idefics2 were seen in document understanding tasks, where Idefics3 achieved a significant performance boost of 13.7 points on DocVQA.
</p>

<p align="justify">
While Idefics3 shows significant improvement over Idefics2, a comparison with other vision-language models of a similar size, such as InternVL2-8B (which scores 91.6 on DocVQA versus Idefics3’s 87.7), would have been valuable, especially on tasks like OCRBench.
</p>

#### InternVL

<p align="justify">
InternVL is a large-scale vision-language foundation model designed to address the limitations of traditional methods in aligning vision and language models. By scaling up the vision model to 6 billion parameters and progressively aligning it with a LLM, InternVL aims to create a more effective integration of these two domains. One major challenge in this integration is the disparity in parameter scales between vision models and LLMs, as well as the inconsistent representations and inefficient connections that fail to fully capture the rich cross-modal interactions. To address these shortcomings, InternVL introduces a novel architecture based on three key design principles:
</p>
<ol>
<li><b>Parameter-balanced Vision and Language Components</b>: The vision encoder in InternVL is scaled up to 6 billion parameters, complemented by an LLM middleware with 8 billion parameters. The middleware serves as a crucial connection layer, reorganizing visual features based on user commands, thus facilitating a more balanced interaction between the vision and language components.</li>
<li><b>Consistent Representations</b>: To ensure a more cohesive connection between the vision and language models, InternVL utilizes a pretrained multilingual LLaMA model to initialize the middleware. This ensures that the vision encoder is effectively aligned with the LLM, leading to consistent cross-modal representations.</li>
<li><b>Progressive Image-Text Alignment</b>: InternVL employs a two-step learning process for image-text alignment. It begins with contrastive learning on large-scale noisy image-text data and gradually transitions to generative learning on fine-grained data. This progressive alignment ensures a consistent improvement in the model’s performance and adaptability across various tasks.</li>
</ol>

<p align="justify">
Thanks to these innovations, InternVL can seamlessly integrate with other large language models, enhancing its versatility and capacity to handle complex vision-language tasks.
</p>

{{< figure align=center alt="InternVL1 architecture overview" src="/imgs/vlms/internvl1_architecture.png" width=70% caption="Figure 12. Overview of InternVL1 model components [16]">}}

<p align="justify">
Figure 12 illustrates the overall architecture of InternVL, highlighting its key components and design. The model's large-scale vision encoder, named InternViT-6B, is based on a vanilla vision transformer. The optimal configuration of InternViT-6B was determined through an extensive hyperparameter search, evaluating different depths, head dimensions, and MLP ratios to maximize its performance. The language middleware, called QLLaMA, is based on a multilingual LLaMA model. This middleware is enhanced with 96 additional learnable queries and cross-attention layers, adding over 1 billion parameters. These enhancements allow QLLaMA to effectively integrate visual elements into the language model, creating a smooth interaction between the vision and language components. By flexibly combining InternViT-6B and QLLaMA, InternVL is capable of supporting a wide range of tasks, including vision-specific tasks like image classification, vision-language tasks like image-text retrieval, as well as generative tasks and multimodal dialogs. This versatility makes InternVL highly adaptable for various vision and vision-language applications.
</p>

{{< figure align=center alt="InternVL1 training stages overview" src="/imgs/vlms/internvl1_training.png" width=100% caption="Figure 13. Overview of InternVL1 training stages [16]">}}

<p align="justify">
As illustrated in Figure 13, the training process of InternVL follows a progressive three-stage approach, consisting of vision-language contrastive training, vision-language generative training, and supervised fine-tuning. Each stage utilizes public data from diverse sources, ranging from noisy image-text pairs to high-quality datasets for captions, visual question answering (VQA), and multi-modal dialogues. A comprehensive overview of the data used is provided in the paper, offering detailed insights into the datasets employed.
</p>

<ol>
<li><b>Vision-Language Contrastive Training</b>: This stage leverages approximately 5 billion image-text pairs sourced from publicly available datasets such as LAION-en, LAION-multi, and Wukong. The objective function used here is the same as that of CLIP, enabling the model to learn strong image-text alignments from large-scale noisy data.</li>
<li><b>Vision-Language Generative Training</b>: In this phase, InternViT-6B (the vision encoder) is connected to QLLaMA (the language middleware), but both components remain frozen. Only the added learnable queries and cross-attention layers are trained using around 1 billion high-quality samples. The training employs the BLIP-2 loss function, which consists of three key components: image-text contrastive (ITC) loss, image-text matching (ITM) loss, and image-grounded text generation (ITG) loss, ensuring a robust generative capability for the model.</li>
<li><b>Supervised Fine-tuning</b>: This final stage demonstrates InternVL's effectiveness in creating multi-modal dialogue systems. It involves connecting InternVL to a larger LLM, such as Vicuna-13B, via an MLP layer. The fine-tuning process uses 4 million instruction data samples to train the model, yielding strong performance even when the LLM decoder remains frozen.</li>
</ol>

<p align="justify">
Through this progressive training strategy, InternVL achieves robust performance across various vision-language tasks, from contrastive learning to multi-modal dialogues.
</p>

<p align="justify">
<b>InternVL1.5</b> introduces three key improvements over its predecessor, InternVL1:
</p>
<ol>
<li><b>Enhanced Vision Encoder</b>: A continuous learning strategy is employed to improve the visual understanding capabilities of the InternViT-6B model. This enables stronger and more accurate visual representation.</li>
<li><b>Dynamic High-Resolution Image Processing</b>: The model now divides images into tiles sized 448x448 based on their aspect ratio and resolution, allowing for better handling of high-resolution images while maintaining efficiency.</li>
<li><b>High-Quality Bilingual Dataset</b>: A comprehensive, high-quality bilingual dataset was collected, featuring common scenes and document images. This significantly improves the model's OCR-related capabilities, making it more adept at understanding text in visual data.</li>
</ol>
<p align="justify">
Additionally, the architecture has evolved. Instead of the middleware used in InternVL1, the authors implemented a ViT-MLP-LLM structure. This is combined with Pixel Shuffle to reduce the number of visual tokens and uses an MLP Projector as a connection mechanism, optimizing performance and efficiency.
</p>

{{< figure align=center alt="InternVL1.5 architecture and image slicing" src="/imgs/vlms/internvl1_5.png" width=100% caption="Figure 14. Overview of InternVL1.5 architecture and image slicing [17]">}}

<p align="justify">
To enhance scalability for high-resolution images, InternVL1.5 employs a pixel shuffle operation that reduces the number of visual tokens to one-quarter of the original. As a result, a 448x448 image is represented by just 256 visual tokens, significantly improving processing efficiency. The multilingual dataset used for training was constructed from openly available datasets tailored for various tasks, which were carefully filtered to ensure quality. For more detailed information on the specific datasets used for both pretraining and fine-tuning, refer to the tables provided in the paper. InternVL1.5 particularly excels in OCR-related tasks and TextVQA benchmarks, outperforming even proprietary models such as GPT-4V and Claude-3 Opus. This demonstrates the model's superior capabilities in understanding and interpreting visual-textual information.
</p>

<p align="justify">
<b>InternVL2</b> is the most current iteration of the model family that ranges in size from 1 billion to 108 billion parameters. While there is no official paper available yet, it is described as being on par with current commercial closed-source models in terms of capabilities. The model is built on several key design principles:
</p>
<ol>
<li><b>Progressive Alignment with LLMs</b>: InternVL2 employs a progressive alignment training strategy, making it the first vision foundation model to be natively aligned with large language models (LLMs). This training strategy scales the model from small to large sizes, while the data used for training evolves from coarse to fine detail. This approach results in more cost-effective training while delivering excellent performance.</li>
<li><b>Multitask Output</b>: The model supports a variety of output formats, including images, bounding boxes, and masks. By linking the model with task-specific decoders, it can generalize across hundreds of vision-language tasks, making it highly versatile.</li>
</ol>

<p align="justify">
The pretraining process extends the stage 1 datasets used for InternVL1.5 with data from diverse sources such as Testbank, OmniCorpus, Kaptest, and more. The stage 2 training uses the same data as InternVL1.5. For more detailed information on the architectural differences, the codebase is available for review (I was to lazy to explore it myself).
</p>

#### Qwen2-VL

<p align="justify">
Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which allows the model to dynamically process images of varying resolutions by converting them into different numbers of visual tokens. The goal of this innovation is to achieve more efficient and accurate visual representations. In addition, the model incorporates Multimodal Rotary Position Embedding (M-RoPE), designed to facilitate the integration of positional information across multiple modalities such as text, images, and videos. The Qwen2-VL series is available in three model sizes: 2 billion, 7 billion, and 72 billion parameters. All these models utilize a 675 million parameter Vision Transformer (ViT) across the various-sized Qwen2 LLMs.
</p>

{{< figure align=center alt="Qwen2-VL architecture overview" src="/imgs/vlms/qwen2vl.png" width=100% caption="Figure 15. Overview of Qwen2-VL architecture [19]">}}

<p align="justify">
The Qwen2-VL model introduces two key architectural enhancements: Naive Dynamic Resolution and Multimodal Rotary Position Embedding (M-RoPE).
</p>

<p align="justify">
The <b>Naive Dynamic Resolution</b> mechanism enables the model to process images of varying sizes by dynamically converting them into a variable number of visual tokens. It removes the original absolute position embeddings of the Vision Transformer (ViT) and instead introduces 2D-RoPE, which is capable of capturing two-dimensional positional information from images. To further streamline image processing, an MLP layer is employed to compress groups of 2x2 tokens into a single token. For example, an image with a resolution of 224x224 is compressed into 66 tokens, making the process more efficient.
</p>

<p align="justify">
The second enhancement, <b>Multimodal Rotary Position Embedding</b> (M-RoPE), is designed to effectively model the positional information of multimodal inputs by decomposing the original rotary embeddings into three distinct components: temporal, height, and width. For text inputs, these components share identical position IDs, making the mechanism functionally equivalent to the standard 1D-RoPE. When processing images, the temporal IDs of visual tokens remain constant, while the height and width components are assigned distinct IDs based on their positions in the image. For videos, the temporal ID increases with each frame, while the width and height components follow the same assignment pattern as in the case of images. This innovative approach not only improves the modeling of positional information but also reduces the reliance on position IDs, allowing the model to generalize to longer sequences during inference. Together, these enhancements make the Qwen2-VL model more efficient and effective in handling multimodal data, with improved extrapolation capabilities for images and videos.
</p>

{{< figure align=center alt="Multimodal Rotaty Position Embeddings" src="/imgs/vlms/mrope.png" width=100% caption="Figure 16. Overview of Multimodal Rotary Position Embeddings [19]">}}


#### MiniCPM-V 

<p align="justify">
Despite the advancements in Vision-Language Models (VLMs), several challenges still prevent their widespread use in real-world applications. One of the most significant obstacles is the high cost associated with running these large models. Typically, VLMs need to be deployed on high-performance cloud servers, which significantly limits their application in mobile, offline, or privacy-sensitive environments. Addressing these challenges, the MiniCPM model family offers a promising solution. These models demonstrate strong performance on general benchmarks, particularly excelling in Optical Character Recognition (OCR) tasks. Furthermore, they provide multilingual support for over 30 languages and, importantly, can be run on mobile devices, expanding their usability to a broader range of applications.
</p>

{{< figure align=center alt="MiniCPM-V-2.5 model architecture" src="/imgs/vlms/minicpm.png" width=100% caption="Figure 11. MiniCPM-V-2.5 model architecture [20]">}}

<p align="justify">
The model architecture consists of three key modules: a visual encoder, a compression layer, and a large language model (LLM). The input image is encoded by a SigLIP SoViT-400m/14, using the adaptive visual encoding approach proposed by LLaVA-UHD. The compression layer has a perceiver resampler structure with a single-layer cross-attention mechanism. Once compressed, the visual tokens, together with the text input, are fed into the LLM for conditional text generation.
</p>
<p align="justify">
The process follows a series of steps: image partitioning, slice encoding (with resizing to match the ViT input size), and token compression. Specifically, each image slice is represented by 1024 tokens, which are then compressed using the cross-attention layer to 64 or 96 tokens. Training this model involves three phases:
</p>
<ol>
<li><b>Pretraining</b>
    <ul>
        <li><b>Stage 1</b>: Train only the compression layer that connects the visual encoder to the LLM, keeping all other parameters frozen.</li>
        <li><b>Stage 2</b>: Extend the input resolution of the visual encoder from 224x224 to 448x448.</li>
        <li><b>Stage 3</b>: Train the visual encoder using the adaptive visual encoding strategy.</li>
    </ul>
</li>
<li><b>Supervised Fine-Tuning</b>: In this phase, the model undergoes fine-tuning on high-quality visual question-answering datasets, labeled by humans or strong models like GPT-4. During this stage, all parameters are unlocked and trained together.</li>
<li><b>Trustworthiness and Hallucination Reduction</b>: This phase utilizes the RLAIF-V framework, which is designed to enhance the model’s trustworthiness and minimize hallucinations. The key here is gathering scalable, high-quality feedback from open-source models to conduct Direct Preference Optimization (DPO). Figure 12 outlines the three steps involved in this method: Response Generation, Feedback Collection, and DPO.</li>
</ol>

{{< figure align=center alt="RLAIF-V framework for hallucination reduction" src="/imgs/vlms/rlaif.png" width=100% caption="Figure 12. RLAIF-V framework for hallucination reduction [20]">}}

<p align="justify">
For more information on the datasets used, the methods for improving the quality of training data, or the advancements in on-device deployments, please refer to the detailed discussions in the paper.
The latest model in the MiniCPM-V series, MiniCPM-V 2.6, offers significant improvements over its predecessor, MiniCPM-V 2.5. However, as of now, there is no paper available that outlines the architecture, datasets, or training procedure for this latest version. Despite this, the model has achieved an impressive score of 65.2 on OpenCompass, a remarkable performance for a model of its size. It even outperforms well-known models like GPT-4o, showcasing its advanced capabilities.
</p>


## Training 

<p align="justify">
Scaling is often key to improving the performance of Visual Language Models (VLMs). However, the vast amount of computational resources required to train large-scale models is out of reach for many companies and academic labs. The cost and infrastructure needed to access this level of compute create barriers to participation in this rapidly advancing field.
Recently, though, researchers have discovered a way to overcome these challenges: a well-designed data curation pipeline. By carefully selecting and curating the training data, it is possible to surpass the typical scaling laws that have dominated model development. In this section, we’ll explore how crucial high-quality data is in training VLMs and the recipes used to create datasets that drive superior performance.
</p>

{{< figure align=center alt="Important training decisions to make" src="/imgs/vlms/training.png" width=100% caption="Figure X. Important decisions to make when training VLMs [2]">}}

<p align="justify">
We’ll explore the crucial role of data in Visual Language Model (VLM) training, focusing on the recipes used to create high-quality datasets (Figure X). Additionally, we’ll discuss grounding, ensuring the model accurately links visual content to text, along with alignment to human preferences, and provide tips for selecting the right model architecture for your specific use case. 
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
Visual Language Models (VLMs) are powerful tools, but they still face significant challenges when it comes to accurately understanding and grounding text prompts. These models sometimes misunderstand or misrepresent the instructions given to them, leading to either ignoring important aspects of the prompt or hallucinating elements that aren’t even part of it.
One of the main issues is their difficulty in comprehending relationships and specific attributes. For example, understanding spatial relations like whether an object is on the left or right, interpreting negations, counting objects, or recognizing attributes such as color and texture are areas where models can stumble. These problems underscore the need for more robust techniques that ensure VLMs are grounded and capable of faithfully interpreting prompts.
Currently, no single solution can resolve these challenges, and this remains an active area of research. However, some methods and tricks have been developed to enhance grounding performance in certain scenarios.
</p>
<p align="justify">
<b>Bounding box annotations</b> offer one such solution, helping VLMs connect visual elements to their corresponding textual descriptions. This method incorporates box regression and Intersection over Union (IoU) loss, which aids in locating and aligning visual concepts with their textual counterparts. Existing datasets like COCO can be employed for this purpose, or detection models can be leveraged to assign bounding boxes to caption nouns. This step significantly improves the model's ability to understand where specific objects are in the image, ensuring a tighter correspondence between what is described and what is visualized.
</p>
<p align="justify">
<b>Negative captioning</b>, often used in contrastive learning objectives, is another technique that helps address the problem of grounding. Negative samples are extensively utilized to prevent model collapse, enhance generalization, and improve the learning of discriminative features. Applying similar techniques to VLMs can mitigate issues like misunderstanding the prompt or hallucination. Results from benchmarks such as <a href="https://github.com/mertyg/vision-language-models-are-bows">ARO</a> have demonstrated the effectiveness of this method, showing its potential to correct misinterpretations and improve overall model performance.
</p>

### Alignment

<p align="justify">
To improve multimodal chat capabilities in Vision-Language Models (VLMs), several techniques such as instruction fine-tuning, Reinforcement Learning from Human Feedback (RLHF), and in-context learning are applied. These approaches are designed to align model outputs with the desired responses and enhance the models' performance.
</p>

<p align="justify">
<b>Instruction-tuning</b> is the process of fine-tuning a VLM on supervised datasets that contain instructions, text, image inputs, and the expected responses. While these datasets are typically much smaller than the vast pretraining datasets—ranging from 100 to 100,000 samples—they are crucial for helping the model learn how to interpret and respond to multimodal inputs effectively.
</p>
<p align="justify">
<b>Reinforcement Learning from Human Feedback</b> (RLHF) is another technique aimed at aligning model outputs with human preferences. In this approach, a reward model is trained to assess and prioritize responses based on how well they match human feedback. This helps in guiding the model toward generating outputs that are more in line with user expectations.
</p>
<p align="justify">
Similar to in-context learning in text-based models, in-context learning can also be applied to VLMs. By providing a set of instruction-image-answer examples in the model’s context, the model can learn to generalize and follow instructions more effectively in multimodal scenarios. For instance, if examples include similar instructions with different images or images that follow a sequence but differ in the accompanying instructions, the model can still successfully produce appropriate responses.
</p>
<p align="justify">
A key dataset used for such training is the MIMIC-IT dataset, which contains 2.8 million in-context instruction-image-answer tuples. These tuples are relevant to the test example in one of three ways: either the instructions are similar but the images are different, the images are the same but the instructions vary, or the images follow a sequential order while the instructions differ. This setup helps the model learn to interpret both the instructions and images in context, improving its ability to respond appropriately to multimodal inputs.
</p>


### Improving text-rich Image Understanding

<p align="justify">
- lets improve OCR 
</p>

### Parameter-Efficient Fine-Tuning 

<p align="justify">
As vision-language models (VLMs) continue to grow in size, fine-tuning all their parameters for each specific downstream task has become increasingly impractical due to computational constraints. To overcome these challenges, researchers have developed Parameter-Efficient Finetuning (PEFT) methods. Instead of adjusting the entire set of parameters in a model, PEFT methods focus on fine-tuning only a subset of parameters, making the process more efficient while still allowing the model to adapt to new tasks.
These PEFT methods can be categorized into four main groups: Low-Rank Adapters (LoRA)-based methods, Prompt-based methods, Adapter-based methods, Mapping-based methods.
</p>

<p align="justify">
<b>LoRA-based methods</b>: very popular method which can be applied to both pure language and vision-language models. Instead of fine-tuning all parameters, LoRA focuses on adjusting only a smaller part of them, by inserting a small matrix inside the model, which is then trained to learn new information while keeping the main model mostly the same. This is not only much cheaper but also faster than a full fine-tuning. In addition you can train different LoRA adapters for different tasks and switch them in and out without changing the base model. Several variants of LoRA have been developed to enhance its functionality and efficiency, like QLoRA, VeRA or DoRA.
</p>
<p align="justify">
<b>Prompt-based methods</b>: Context Optimization (CoOp), which is a technique designed to adapt pre-trained VLMs for downstream image recognition tasks, eliminating the need for manual prompt engineering. It does that by optimizing the context of the prompt using learnable vectors during the training process. Experiments from 11 datasets indicate that CoOp outperforms handcrafted prompts and linear probe models in few shot learning. Another interesting method is called Visual Prompt Tuning (VPT) that adapts Transformers models in vision by introducing a small amount of trainable parameters in the input space (task-specific learnable prompts). 
</p>

{{< figure align=center alt="Overview of prompt-based methods to improve vision language models" src="/imgs/vlms/coop_vpt.png" width=100% caption="Figure X. Overview of prompt-based methods for finetuning VLMs [14] & [15]">}}

<p align="justify">
<b>Adapter-based methods</b>: Adapters are new modules added between layers of a pre-trained network. Examples of this: CLIP-Adapter, VL-adapter and Llama-Adapter V2. CLIP-Adapter: appends a small number of additional learnable bottleneck linear layers to CLIP’s language and image branches while keep the original CLIP backbone frozen during few-shot fine-tuning. deal with over-fitting and improve the robustness of CLIP-Adapter, we further adopt residual connections to dynamically blend the fine-tuned knowledge with the original knowledge from CLIP’s backbone
</p>
<p align="justify">
<b>Mapping-based methods</b>:
</p>




## Evaluation & Benchmarks 

<p align="justify">
</p>

{{< figure align=center alt="Overview of evaluation methods for vision language models" src="/imgs/vlms/evaluation.png" width=90% caption="Figure X. Overview of evaluation methods for VLMs [2]">}}



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

[[14]](https://arxiv.org/pdf/2109.01134) Zhou et al. "Learning to Prompt for Vision-Language Models" (2022)

[[15]](https://arxiv.org/pdf/2203.12119) Jia el al. "Visual Prompt Tuning" (2022)

[[16]](https://arxiv.org/pdf/2312.14238) Chen el al. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (2024)

[[17]](https://arxiv.org/pdf/2404.16821) Chen el al. "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open Source Suites" (2024)

[[18]](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) Chen el al. "InternVL2: Better than the Best - Expanding Performance Boundaries of Open Source Multimodal Models with the Progressive Scaling Strategy" (2024)

[[19]](https://arxiv.org/pdf/2409.12191) Wang el al. "Qwen2-VL: Enhancing Vision-Language Models Perception of the World at Any Resolution" (2024)

[[20]](https://arxiv.org/pdf/2408.01800) Yao el al. "MiniCPM-V: A GPT-4V Level MLLM on Your Phone" (2024)