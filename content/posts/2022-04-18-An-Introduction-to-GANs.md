---
layout: post
title: An Introduction to Generative Adversarial Networks
date: 2022-04-18 10:00:00 +0200
author: Johann Gerberding
summary: A short introduction to the world of Generative Adversarial Networks including basic concepts, popular model architectures and evaluation metrics.
include_toc: true
showToc: true
math: true
draft: true
---


## Introduction & Concepts

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs). Most GANs consist of two components: <b>generator</b> and <b>discriminator</b>.
</p>

<p align="justify">
<b>Discriminator:</b> It is a classifier with two classes: <i>real</i> or <i>fake</i>. The input can be anything from images or videos to text or audio. It calculates the probabilities $p(y=real|x)$ and $p(y=fake|x)$.
</p>

<p align="justify">
<b>Generator:</b> It represents different classes in general, not only distinguish them. It has to figure out $p(x|y)$: the probability that, given you generated y=dog, the resulting image $x$ is the one generated. The output space of possible dog images is huge, which makes this task very challenging and harder than the discriminator task. Typically you have the generator to take multiple steps to improve itself before training the discriminator again.
</p>

<p align="justify">
The input to the generator is called <b>noise vector</b> $z$, with the role of making sure the images generated for the same class don't look too similar (like a random seed). It is often generated randomly by sampling random numbers either between 0 and 1 uniformly or from a normal distribution. The size of the vector is often 100 or a power of 2, so that it is large enough to contain a lot of combinations. Another popular concepts that people use to tune their outputs is called the <i>truncation trick</i>, which is a way of trading off fidelity (quality) and diversity (variety) in the samples. This is achieved by resampling the noise vector multiple times until it falls within some bounds of the normal distributions. By tuning the boundary values you can control fidelity vs. diversity.
</p>

## Popular Algorithms

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### Wasserstein GANs

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### StyleGAN 

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### ProGAN

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>


## Evaluation

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### Inception Score (IS)

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### Fréchet Inception Distance (FID)

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

### Precision & Recall

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

## Shortcomings

<p align="justify">
This post will give a short introduction to the basic concepts of Generative Adversarial Networks (GANs).
</p>

## Summary

<p align="justify">
In this post we looked at three combined RL methods, A2C, PPO and AlphaGoZero. As Alpha(Go)Zero is a more specialized method for games, PPO and A2C are more generally applicable. I hope I was able to give you an understandable insight into the three algorithms and their most important components. In the next post of this series I am going to dive a bit deeper into model-based RL and imitation learning. Until then, stay tuned and healthy!
</p>

## References

[[1]](https://arxiv.org/abs/1602.01783) Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning" (2016).
