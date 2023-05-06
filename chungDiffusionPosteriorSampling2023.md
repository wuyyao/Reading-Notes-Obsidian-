# Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model  
<cite>* Authors: [[Yinhuai Wang]], [[Jiwen Yu]], [[Jian Zhang]]</cite>

* Date: [[2022-12-07]]

* URL: [http://arxiv.org/abs/2212.00490](http://arxiv.org/abs/2212.00490)  

* DOI: [10.48550/arXiv.2212.00490](https://doi.org/10.48550/arXiv.2212.00490)  

* Tags: #Computer-Science---Computer-Vision-and-Pattern-Recognition

* Cite key: undefined

* [Local library](zotero://select/items/1_RFXIKYBK)  

* PDF Attachments
	- [arXiv Fulltext PDF](zotero://open-pdf/library/items/DNTLKQN7)   

***

## Highlights and Annotations

## 1. 介绍

### 1.1 Abstract

·现有的大多数工作是在无噪声环境下求解简单的线性反问题，这在很大程度上低估了实际问题的复杂性。我们通过对**后验采样的近似**，扩展了扩散求解器来有效地处理一般的**含噪(非线性)线性反问题**。得到的后验采样方案是一个混合版本的扩散采样，带有流形约束梯度，没有严格的测量一致性投影步骤，与以前的研究相比，在噪声环境中产生了更理想的生成路径。该方法证明了扩散模型可以包含高斯和泊松等各种测量噪声统计，并且可以有效地处理傅里叶相位恢复和非均匀去模糊等噪声非线性逆问题。

### 1.2 Introduction

·1）当测量中存在噪声时，投影型方法明显会失败，因为噪声通常在生成过程中由于反问题的不适定性而被放大；2）测量过程是非线性的。

·我们设计了一种方法，通过新的近似方法来避免扩散模型的后验采样困难。具体来说，我们的方法可以有效地处理**高斯和泊松**测量噪声。

·MCG方法是该方法在测量无噪声情况下的一个特例。本文提出的方法在噪声环境下更能产生理想的样本路径。

·所提方法完全运行在图像域而非谱域上，从而能避免SVD的计算。

## 2. Background

### 2.1 基于分数的扩散模型

·数据去噪过程的Ito SDE形式：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/1.png?raw=true)
·逆向SDE过程形式如下：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/2.png?raw=true)
·得分函数∇xtlogpt(xt)由经过去噪得分匹配训练的神经网络sθ逼近：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/3.png?raw=true)

### 2.2 扩散模型用于逆问题求解

·利用扩散模型作为先验，可得到从后验分布采样的逆向扩散采样器：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/4.png?raw=true)其中，需要计算得分函数∇xt log pt(xt)和似然项∇xt log pt(y|xt)。前一项可以使用预训练的sθ∗得到，后一项依赖于时间t是不易计算的，y只与x0之间存在显式依赖关系。

·前向模型的一般形式可以描述为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/5.png?raw=true)
其中A(·) : Rd → Rn是前向测量算子，n是测量噪声。在高斯白噪声的情况下，n ∼ N (0, σ2I)，于是有p(y|x0) ∼ N (y|A(x0), σ2I)，但是在y与xt之间没有显式依赖。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/6.png?raw=true)

·为了避免直接使用似然项，交替投影到测量子空间是一种广泛使用的策略。可以忽略式4中的似然项，先对式2进行无条件更新，然后在假设n接近于0的情况下进行一个投影步骤，使得满足测量一致性。  
另一种方法解决线性逆问题是利用一个近似：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/7.png?raw=true)
当n假设为方差是σ2的高斯噪声。然而，该方程只在t=0时是正确的，在生成过程中实际使用的所有其他噪声水平下都是错误的，这种错误可以通过在t→T时假设更高的噪声水平来抵消，有![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/8.png?raw=true)
![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/8.png?raw=true)
其中{γt}是超参数。  
这两种方法目的都是在测量值给定的情况下进行后验采样，并且用于无噪声反问题。但是1）它们没有处理测量噪声；2）这些方法处理非线性反问题是无法工作或者不易实现的。

## 3. 扩散后验采样(DPS)

### 3.1 似然项的近似

·利用测量模型p(y|x0)分解p(y|xt)如下：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/9.png?raw=true)
分解出来存在p(x0|xt)是不易求解的。然而，对于VP-SDE或DDPM等扩散模型，前向扩散可以简单表示为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/10.png?raw=true)
从而我们可以获得如命题1所示的后验均值的特殊表示。

·命题1：对于VP-SDE或DDPM采样，p(x0|xt)有唯一的后验均值：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/11.png?raw=true)
将∇xt log p(xt)替换成分数估计sθ∗(xt)可得后验均值的近似为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/12.png?raw=true)
后验均值x0可以在中间步骤中有效计算，所以考虑为p(y|xt)提供一个易于处理的近似，使得我们可以用替代函数来最大化似然收益近似后验抽样。具体来说，将p(y|xt) = Ex0~p (x0|xt)[p(y|x0)]使用以下近似：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/13.png?raw=true)将p(y|x0)关于后验分布的外期望替换为x0的内期望。这类估计与Jensen’s不等式相关，因此我们需要以下定义来量化逼近误差。

·定义1（Jensen gap）：设x是服从p(x)分布的随机变量，对于某些函数f可能是凸函数也可能不是凸函数，Jensen gap定义为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/14.png?raw=true)其中p(x)取期望。

·定理1：对于给定的测量模型(6)，当n~N(0, σ2I)有![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/15.png?raw=true)
其中逼近误差可以用Jensen gap来量化，其上界为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/16.png?raw=true)其中 ‖∇xA(x)‖ := maxx ‖∇xA(x)‖ 以及m1 := ∫ ‖x0 − x^0‖ p(x0|xt)dx0

·‖∇xA(x)‖ 在大多数反问题中是有限的，这与反问题的不适定性不同，不适定性指的是逆算子A-1的无界性。相应地，若m1也是有限的，则当σ → ∞时，定理1中的Jensen gap可以趋近于0，这表明近似误差随着测量噪声的增大而减小。

·利用定理1的结果，我们可以使用对数似然的近似梯度![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/17.png?raw=true)
后者的分布是可分析易处理的，因为测量分布是给定的。

### 3.2 测量的模型依赖似然

·将扩散后验抽样用于高斯噪声，似然函数形式为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/18.png?raw=true)
其中n代表测量值y的维度。通过对p(y|xt)关于xt进行求导，利用定理1和公式15可得![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/19.png?raw=true)
其中明显标记了xˆ0 := xˆ0(xt)来强调x^0是xt的函数，取梯度∇xt是通过网络进行反向传播，将定理1至5的结果与训练好的得分函数相加，可得结论：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/20.png?raw=true)
其中，ρ定义为1/σ2作为步长。

·将扩散后验采样用于泊松噪声，在独立同分布假设下，泊松测量的似然函数为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/21.png?raw=true)
其中，j为测量仓索引。在大多数测量值不太小的情况下，模型可以用精度非常高的高斯分布来近似，即![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/22.png?raw=true)
其中使用散粒噪声模型[A(x0)]j的标准近似得到最后一个方程。类似高斯情形，通过微分并利用定理1，我们有![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/23.png?raw=true)
其中‖a‖2Λ定义为aTΛa ，已经包括了ρ来定义高斯情况下的步长。

·针对每个噪声模型的策略总结如下：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/24.png?raw=true)![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/25.png?raw=true)

·DPS的几何结构与MCG之间的联系：MCG的论文中，通过式16在更新步骤后额外执行到测量子空间的投影，可以认为是对数据一致性偏差的修正。在这篇论文的扩散模型中，通过sθ∗的单个去噪步骤对应于数据流形的正交投影，并且梯度步骤∇xi ‖y − A(xˆ0)‖22是与当前流形相切的步骤。  
对于含噪反问题，MCG论文在每一步梯度后对测量子空间进行投影时，由于过分加强仅使用于无噪测量的数据一致性，样本可能会从流形上脱落，从而得到错误的解。本论文提出的方法不需要在测量子空间上进行投影，所以不存在上述缺点。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungDiffusionPosteriorSampling2023/26.png?raw=true)

## 4. Experiments

·实验数据集：FFHQ 256×256 以及 Imagenet 256×256  
实验评估指标：FID 和 LPIPS  
实验的线性反问题：SR、Inpaint、Gaussian deblur、Motion deblur  
实验的非线性反问题：Phase retrieval和Non-uniform deblur

·局限性：1）继承了基于扩散模型方法的缺点，所提出的方法相对比较慢，但是仍然比基于GAN的优化方法更快，因为不需要对网络本身进行微调。2）该方法倾向于保留图像的高频细节（例如胡须、头发和纹理等）。3）相位恢复的重建质量不如线性反问题和非线性去模糊那么稳健。