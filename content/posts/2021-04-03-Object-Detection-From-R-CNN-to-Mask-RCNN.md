---
title: Object Detection - From R-CNN to Mask R-CNN
date: 2021-04-03 10:23:16 +0200
author: Johann Gerberding
summary: Two-Stage Object Detection Models.
include_toc: true
showToc: true
math: true
tags: ["deep-learning", "computer-vision", "convolutional-neural-networks"]
---

## Introduction

<p align="justify">
In the following weeks (or months) I am going to take a deep dive into Deep Learning based Object Detection models. My goal is to create a series of posts regarding different approaches and popular architectures for this task. In this post I'll start with the description of the R-CNN ("Region-based Convolutional Neural Networks") model family, their emergence and central ideas. Nowadays there exist much more accurate and efficient architectures but I think it's a good starting point for such a series. Since the R-CNN based models are so called two-stage approaches, I'll go over a few popular one-stage architectures like YOLO in the next post as well. I'll describe in a minute what exactly this means and what the key differences are. Furthermore, following these two posts, I would like to discuss more current architectures that represent the state-of-the-art in the most recognized benchmarks like <i>EfficientDet</i> or new Transformer-based approaches like <i>DETR</i>.
</p>

<p align="justify">
As mentioned before, we can broadly distinguish between one and two-stage detection frameworks. A typical two-stage pipeline consists of an initial category-independent region-proposal stage followed by the feature extraction and classification. This allows for a high localization and recognition accuracy. In contrast single-stage object detectors do not require prior proposals which makes them faster but less accurate. But more on that in the next post. Now let's start with the R-CNN models.
</p>

## R-CNN

<p align="justify">
Ross Girshick et al. presented their approach called "Region-based Convolutional Neural Networks" (R-CNN) in 2014. It was one of the first methods based on deep convolutional networks (CNNs) besides e.g. <i>Overfeat</i>. The main idea is to tackle the problem of Object Detection in several successive steps, as shown in Figure 1.
</p>

{{< figure align=center alt="R-CNN model workflow" src="/imgs/object_detection_1/RCNN.png" width=100% caption="Figure 1. R-CNN model workflow [1]">}}

<p align="justify">
It was an important contribution to the Computer Vision community because of the significant improvement of the state-of-the-art at that time (mAP improvement of more than 30% on PASCAL VOC). Moreover the structure as well as the workflow of the framework are pretty straightforward:
</p>

1. Extract around 2000 of category-independent bounding box object region candidates per image using [selective search](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)
2. Each region candidate gets warped to have a fixed size of 227x227 pixels which is required by the CNN (you can detailed information on how this is done in the Appendix of the paper)
3. Extract the features of each candidate with a CNN (AlexNet)
4. Classification of each region with class-specific Support Vector Machines (SVM)
5. Bounding box regression based on the predicted class, predict bounding box offsets (boosts up mAP by 3-4 points)

<p align="justify">
The overall training procedure is a stepwise process and requires a lot of work. First you have to pretrain your CNN classification on ImageNet. In the next step you have to get rid of the last classification layer and insert a new one with $K+1$ classes ($K$ = number of classes ;+1 for background). Start finetuning this network using warped proposal windows. It is very important for training that you reduce your learning rate when finetuning (0.01 for ImageNet and 0.001 for finetuning). In the training process all proposals with an Intersection over Union (IoU) >= 0.5 are considered positive samples. The mini-batch size here was 128, consisting of 32 positive and 96 negative boxes, so its biased towards the positives (selective search produces much more negatives than positives). After finetuning your CNN we start building the class-specific binary SVMs. Here a the authors used grid search to choose the IoU threshold of 0.3. In addition to speed up the process they used hard negativ mining. If you want the details on why it is done this way you can look it up in the Appendix of the paper. The last step of the training procedure is the creation of class-specific bounding box regressors, which output bbox offsets. To train these only proposals with an IoU >= 0.6 are used.
</p>

<p align="justify">
Now let's dive a bit deeper into <b>Bounding Box Regression</b>. The offsets get calculated based on the features after $pool_{5}$ layer of each proposal $\mathbf{p}$. The regressor is build to learn scale-invariant transformations between the centers and a log-scale transformation between widths and heights. This is illustrated down below.
</p>

{{< figure align=center alt="Bounding Box Regression" src="/imgs/object_detection_1/RCNN-bbox-regression.png" width=80% caption="Figure 2. R-CNN - Bounding Box Regression">}}

$$
\hat{g}\_{x} = p\_{w}d\_{x}(\mathbf{p}) + p\_{x} \\
\hat{g}\_{y} = p\_{h}d\_{y}(\mathbf{p}) + p\_{y} \\
\hat{g}\_{w} = p\_{w}e^{d\_{w}(\mathbf{p})} \\
\hat{g}\_{h} = p\_{h}e^{d\_{h}(\mathbf{p})}
$$

$\mathbf{p}$ represents the predicted bbox information $(p\_{x}, p\_{y}, p\_{w}, p\_{h})$ whereas $\mathbf{g}$ contains the ground truth values $(g\_{x}, g\_{y}, g\_{w}, g\_{h})$. The targets to learn are the following:

$$
t\_{x} = (g\_{x} - p\_{x}) / p\_{w} \\
t\_{y} = (g\_{y} - p\_{y}) / p\_{h} \\
t\_{w} = \log (g\_{w} / p\_{w}) \\
t\_{h} = \log (g\_{h} / p\_{h})
$$

<p align="justify">
A regression model can solve the problem by minimizing the Sum of Squared Error Loss with regularization (ridge regression):
</p>

$$
L\_{reg} = \sum_{i \in \{ x, y, w, h \}} (t\_{i} - d\_{i}(\mathbf{p}))^{2} + \lambda \| \mathbf{w} \|^{2}
$$

<p align="justify">
The regularization term ($\lambda$ = 1000) is critical and the authors picked it by using cross validation. One benefit of the application of these transformations is that alle the box correction functions $d_{i}(p)$ where $i \in \{ x, y, w, h \}$ can take any value $[- \infty, + \infty]$.
</p>

**Shortcomings:**
* training is a multi-stage pipeline
* training is expensive in space and time
* speed bottleneck due to selective search and feature vector generation for every region proposal ($N$ images $*$ 2000)

## Fast R-CNN

<p align="justify">
To overcome these shortcomings described above, Girshick improved the R-CNN training procedure by unifying the three models into one jointly trainable framework called <b>Fast R-CNN</b>. Instead of extracting feature vectors for every object proposal separately, here the entire image gets forward passed through a deep CNN (<a href="https://arxiv.org/pdf/1409.1556.pdf">VGG16</a>)) to produce a convolutional feature map. For each object proposal a <b>Region of Interest (RoI) pooling layer</b> extracts a fixed length feature vector from this feature map. Those feature vectors are then fed into multiple fully-connected layers which finally branch into the object classifier and a bounding box regressor. This intergration leads to a lot of shared computation which speeds up the whole prediction process.
</p>

{{< figure align=center alt="Fast R-CNN model workflow" src="/imgs/object_detection_1/fast-RCNN.png" width=100% caption="Figure 3. Fast R-CNN architecture">}}

<p align="justify">
One new key component of the proposed framework is the RoI pooling layer which is a type of max pooling that allows us to convert the features inside any valid region proposals into a feature map with a fixed window of size H $\times$ W. This is done by dividing the input region into H $\times$ W small grids where every subwindow size is approximately of size [h/H $\times$ w/W]. In each of those grid cells apply regular max pooling.
</p>

{{< figure align=center alt="Region of Interest Pooling" src="/imgs/object_detection_1/roi_pooling.png" width=100% caption="Figure 4. Region of Interest Pooling">}}

<p align="justify">
The training procedure of this framework is somewhat similar to the one of R-CNN. Here they also pre-train a deep CNN on ImageNet and use selective search for proposal generation. After pre-training the last max pooling layer is replaced by a RoI pooling layer which is set to be compatible with the nets first fully connected layer, for VGG16 H = W = 7. The last fully connected layer and softmax are replaced by two sibling layers, one for classification consisting of one fully connected layer followed by a softmax over $K+1$ categories and one for category-specific bounding box regressors. Another major advantage of this approach is the possibility of end-to-end training with a multi-task loss. This loss function sums up the cost of classification and bbox prediction:
</p>

$$
L(p,u,t^{u},v) = L\_{cls}(p,u) + \lambda [u \geq 1] L\_{loc}(t^{u}, v)
$$

Symbol | Description
--- | --- 
$p$ | discrete probability distribution (per RoI) over K+1 classes, $p=(p\_{0}, ..., p\_{K})$ computed by a softmax
$u$ | ground-truth class label, $u = 1, ..., K$ (background $u = 0$)
$t^{u}$ | predicted bounding box offsets, $t^{u} = (t\_{x}^{u},t\_{y}^{u},t\_{w}^{u},t\_{h}^{u})$
$v$ | true bouding box regression targets $v = (v\_{x},v\_{y},v\_{w},v\_{h})$
$\lambda$ | hyperparameter to control the balance between the two losses

The indicator function $[u \geq 1]$ is defined as

$$
[u \geq 1] =
\begin{cases}
    1       & \quad \text{if } u \geq 1\\
    0       & \quad \text{otherwise}
\end{cases}
$$

<p align="justify">
to ignore background classifications. $L_{cls}(p,u) = - \log p_{u}$ is a log loss for the true class $u$. The bounding box loss is defined as followed:
</p>

$$
L_{box}(t^{u}, v) = \sum_{i \in \{x,y,w,h\}} smooth_{L\_{1}} (t\_{i}^{u} - v\_{i})
$$

<p align="justify">
It measures the difference between $t_{i}^{u}$ and $v_{i}$ using a robust smooth $L_{1}$ loss function which is claimed to be less sensitive to outliers than the $L_{2}$ loss.
</p>

$$
L\_{1}^{smooth}(x) =
\begin{cases}
    0.5x^2       & \quad \text{if } |x| < 1\\
    |x| - 0.5      & \quad \text{otherwise}
\end{cases}
$$

<p align="justify">
As mentioned before Fast R-CNN is much faster in training and testing but one speed bottleneck in form of selective search still remains which leads us to the next evolutionary stage of the architecture.
</p>

## Faster R-CNN

<p align="justify">
Here the region proposal step gets integrated into the CNN in form of a so called <b>Region Proposal Network</b> (RPN) that shares conv features with the detection network ("it tells the classification network where to look").
</p>

<p align="justify">
The RPN takes an image of arbitrary size as input and outputs a set of rectangular object proposals and objectness scores (= object vs. background). It is important to keep in mind that the goal here is to share computation, so a set of conv layers is chosen to be also part of the object detection pipeline to extract features.
</p>

{{< figure align=center alt="Faster R-CNN model" src="/imgs/object_detection_1/faster-RCNN.png" width=100% caption="Figure 5. Faster R-CNN architecture (left) and RPN workflow (right)">}}

<p align="justify">
To generate region proposals, a small $n \times n$ spatial window gets slided over the last shareable conv feature map to create lower dimensional feature vectors (256 dims). These reduced feature vectors get fed into two sibling fully connected layers, a bbox regression  layer and a box classification layer. It is implemented using one $n \times n$ convolution and two $1 \times 1$ conv layers. The classification part is implemented as a two class softmax layer. At each sliding window location $k$ regions of various scales and ratios get proposed simultaneously (k = 9). Those proposals are parameterized relative to $k$ reference boxes called <b>anchors</b>. Each anchor is centered at the sliding window position and is associated with a combination of different aspect ratios (1:1, 1:2, 2:1) and scales (128x128, 256x256, 512x512). This multi-scale anchor-based design is a key component for sharing features without extra cost for addressing scales and aspect ratios.
</p>

<p align="justify">
The loss function used to train the RPN is also a multi-task loss which is similar to the one of Fast R-CNN we discussed before:
</p>

$$
L(\{p\_{i}\},\{t\_{i}\}) = \frac{1}{N\_{cls}} \sum L\_{cls} (p\_{i}, p\_{i}^{\ast}) + \lambda \frac{1}{N\_{reg}} \sum p\_{i}^{\ast} L\_{reg} (t\_{i}, t\_{i}^{\ast})
$$

Symbol | Description
--- | --- 
$p_{i}$ | predicted probability that anchor $i$ is an object
$p_{i}^{\ast}$ | ground-truth label (1 or 0)
$t_{i}$ | predicted parameterized coordinates $(x,y,w,h)$
$t_{i}^{\ast}$ | ground-truth parameterized coordinates $(x,y,w,h)$
$\lambda$ | balancing parameter, set to be 10, to balance out $L\_{reg}$ and $L\_{cls}$
$N_{cls}$ | normalization term, set to be the same as the mini batch size (256)
$N_{reg}$ | normalization term, set to be approx. the number of anchor boxes ($\sim$ 2400)

The regression loss is also smooth $L\_{1}$ like in Fast R-CNN. The classification loss can be calculated as followed:

$$
L\_{cls} (p\_{i}, p\_{i}^{\ast}) = -p\_{i}^{\ast} \log p\_{i} - (1 - p\_{i}^{\ast}) \log (1 - p\_{i})
$$

<p align="justify">
For the calculations of $t\_{i}$ and $t\_{i}^{\ast}$ take a look at section 3.1.2 of the <a href="https://arxiv.org/abs/1506.01497">paper</a>. To train the network the authors use a mini batch size of 256 randomly sampled anchors consisting of 128 positives and negatives (padded with negative ones if necessary). All anchors which are beyond the image boundaries are ignored in the training process. For the Fast R-CNN framework integration the authors experimented with different training strategies and chose an alternating one. First they train the RPN and then the Fast R-CNN using the RPN proposals and the trainede backbone. Thereafter they initialize the RPN with the tuned Fast R-CNN network and iterate a couple of times. For detailed evaluation and benchmarks take a look at the paper.
</p>

## Mask R-CNN

<p align="justify">
In the last step of this "evolution", Faster R-CNN was further extended to the task of pixel level instance segmentation, called <b>Mask R-CNN</b>. A third branch is added in parallel to the existing classification and localization branches for predicting binary object masks for every proposed RoI of size $m \times m$.
</p>

{{< figure align=center alt="Mask R-CNN model" src="/imgs/object_detection_1/mask-rcnn.png" width=100% caption="Figure 6. Mask R-CNN architecture">}}

<p align="justify">
The mask branch consists of a fully convolutional network (FCN) which allows each layer to maintain the explicit $m \times m$ object spatial layout without collapsing it into a vector representation that lacks those spatial dimensions. To ensure high quality and precision mask predictions the RoI features have to be well aligned to preserve the explicit per-pixel spatial correspondence. The RoI pooling layer lacks the required precision because of quantization which motivated the authors to develop the so called <b>RoI Align</b> layer. RoI Align avoids quantization and instead uses <a href="https://en.wikipedia.org/wiki/Bilinear_interpolation">Bilinear Interpolation</a> to compute the exact values of the input features at four regularly sampled locations in each RoI bin and aggregate the result (max or avg pooling). This leads to large improvements in mask prediction quality. To get a good understanding of how RoI Align works, take a look at the following <a href="https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193">blogpost</a>.
</p>

<p align="justify">
The authors also demonstrate the generality of the proposed framework by experimenting with different backbones (ResNet, ResNext, FPN, ...) and heads for prediction of bboxes, classification and mask prediction. Moreover, they show that the framework is able to predict keypoints with minimal adjustments. For details on this, take a look at the <a href="https://arxiv.org/abs/1703.06870">paper</a>. For training the multi-task loss of Faster R-CNN for each RoI gets extended by adding a mask loss:
</p>

$$
L\_{mask} = -\frac{1}{m^{2}} \sum_{1 \leq i, j \leq m} [ y\_{ij} \log \hat{y}\_{ij}^{k} + (1-y\_{ij}) \log (1 - \hat{y}\_{ij}^{k}) ]
$$

<p align="justify">
$y_{ij}$ describes the label of a cell (i,j) in the true mask whereas $\hat{y}_{ij}^{k}$ represents the predicted value of cell (i,j). The loss is defined as the average binary cross entropy loss, only including the k-th mask if the region is associated with the ground truth class $k$. The mask branch outputs $K * m^{2}$ binary masks but only the k-th mask contributes to the loss (rest gets ignored). This provides a decoupling between class and mask predictions.
</p>

## Summary

<p align="justify">
To sum everything up and give a broad overview of all architectures covered in this post take a look at the following overview created by Lilian Weng.
</p>

{{< figure align=center alt="R-CNN model family" src="/imgs/object_detection_1/rcnn-family-summary.png" width=100% caption="Figure 7. Overview R-CNN model family">}}

<p align="justify">
For a general overview of the field of Deep Learning based Object Detection I highly recommend <a href="https://arxiv.org/pdf/1809.02165v1.pdf">this survey</a> ("Deep Learning for Generic Object Detection: A Survey") from 2018 which in my opinion gives a great overview for beginners.
</p>

## References

[[1]](https://arxiv.org/pdf/1311.2524.pdf) R. Girshick, J. Donahue, T. Darrell, J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation, 2014.

[[2]](https://arxiv.org/pdf/1504.08083.pdf) Ross Girshick. Fast R-CNN, 2015.

[[3]](https://arxiv.org/pdf/1506.01497.pdf) S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, 2016.

[[4]](https://arxiv.org/pdf/1703.06870.pdf) k. He, G. Gkioxari, P. Dollár, R. Girshick. Mask R-CNN, 2017.