---
title: Imitation Learning  
date: 2022-09-25 10:00:00 +0200
permalink: /:title
author: Johann Gerberding
summary: Introduction to Imitation Learning.
include_toc: true
showToc: true
math: true
draft: true
---

## Introduction

<p align="justify">
As we've learned in a previous post, the goal of Reinforcement Learning is to learn an optimal policy which maximizes the long-term cumulative rewards. Generally many of these methods perform pretty well but in some cases it can be very challenging to learn a even a good policy. This is especially true for environments where the rewards are sparse, e.g. a game where the reward is only received at the end. In such cases it can be very helpful to design a reward function which provide the agent with more frequent rewards. Moreover there are a lot of use cases especially in real world scenarios where it is extremly complicated to design a reward function, e.g. in autonomuous driving.  
</p>

<p align="justify">
Imitation Learning (IL) can be a straightforward and feasible solution for these problems. In IL instead of trying to learn from sparse rewards or complicated and imperfect reward functions, we utilize expert demonstrations which we try to mimic.  
</p>

## Imitation Learning in a Nutshell 

<p align="justify">

</p>

## Types of Imitation Learning 

- Behavioral Cloning 
- Inverse Reinforcement Learning 
- Direct Policy Learning 

## Formal Definition 

<p align="justify">
</p>


## Difference between IL and Offline RL 

<p align="justify">
- very similar, in IL you assume the expert policy is optimal and you try to recover it
- in offline RL the goal is "order out the chaos", find a better policy than you have seen in the data
- offline RL methods must do two things:
	1. stay close to the provided data
	2. maximize reward
- the reward maximization is what is missing in IL    
</p>


## Inverse Reinforcement Learning  

<p align="justify">
- learn the reward function 
- with and without model 
</p>

## Direct Policy Learning  

<p align="justify">

</p>


## Behavior Cloning  

<p align="justify">

- it is really simple and easy to use 
	- very stable (supervised learning)
	- easy to debug and validate 
	- scales well to large datasets 

</p>


## Conditional Imitation Learning  

<p align="justify">
</p>


## References 

Zoltan Lorincz - A brief overview of Imitation Learning (2019) 
https://smartlabai.medium.com/a-brief-overview-of-imitation-learning-8a8a75c44a9c

Yue, Le - Imitation Learning Tutorial (ICML 2018)
https://sites.google.com/view/icml2018-imitation-learning/

Sergey Levine - Imitation Learning vs. Offline Reinforcement Learning 
https://www.youtube.com/watch?v=sVPm7zOrBxM&ab_channel=RAIL
