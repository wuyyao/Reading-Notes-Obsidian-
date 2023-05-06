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

## 1.介绍

### 1.1 Abstract

·“By studying the generative sampling path, here we show that current solvers throw the sample path off the data manifold, and hence the error accumulates.” (Chung 等, 2022, p. 1) 🔤通过对生成式采样路径的研究，我们发现当前求解器将采样路径抛离数据流形，从而导致误差累积。🔤

·”we propose an additional correction term inspired by the manifold constraint, which can be used synergistically with the previous solvers to make the iterations close to the manifold” (Chung 等, 2022, p. 1) 🔤我们提出了一个受**流形约束**启发的额外修正项，该修正项可以与之前的求解器协同使用，使迭代接近流形🔤

### 1.2 Introduction

·“given a pre-trained unconditional score function (i.e. denoiser), solving the reverse stochastic differential equation (SDE) numerically would amount to sampling from the data generating distribution [41]” (Chung 等, 2022, p. 1)扩散模型不需要针对特定问题进行训练，给定一个预训练的无条件得分函数(即去噪器)，数值求解反向随机微分方程(SDE)相当于从数据生成分布中采样。

·“we leverage the denoising result through Tweedie’s formula and show that such denoised samples can be the key to significantly improving the performance of reconstruction using diffusion models across arbitrary linear inverse problems, despite the simplicity in the implementation. Moreover, we theoretically prove that if the score function estimation is globally optimal, the correction term from the manifold constraint enforces the sample path to stay on the plane tangent to the data manifold1, so by combining with the reverse diffusion step, the solution becomes more stable and accurate.” (Chung 等, 2022, p. 2)如果得分函数是全局最优的，那么流行约束的修正项会迫使样本路径停留在与数据流形1相切的平面上，结合反向扩散步骤，解会变得更加稳定和精确。

## 2. Related Works

### 2.1 去噪模型

·连续形式：连续扩散过程x(t)：x(0)~p0(x) = pdata是感兴趣的数据分布，x(0)~p0(x) = pdata是感兴趣的数据分布是不含数据的近似球形高斯分布。那么用Ito SDE表示前向去噪过程：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/1.png?raw=true)
反向SDE过程可以表示为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/2.png?raw=true)在实际应用中，通常最小化以下的去噪得分匹配目标：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/3.png?raw=true)

·离散形式：前向扩散的总体形式为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/4.png?raw=true)离散逆扩散过程总体上可以表示为![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/5.png?raw=true)

### 2.2 逆问题的条件生成模型

·inverse problem的一般形式：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/6.png?raw=true)目标是从关于测量值y的条件分布中产生样本，即p(x | y)。相应地，将得分函数∇xlog pt(x)替换为∇xlopt(x|y)，但是每当条件发生变化时都要重新训练条件得分，这限制了神经网络的泛化能力。最近的条件扩散模型利用无条件得分函数∇xlog pt(x)，但是依赖于基于投影的测量约束来施加条件，应用如下：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/7.png?raw=true)

### 2.3 Tweedie’s formula

·在高斯噪声的情况下，可以通过Tweedie’s formula计算后验期望得到去噪后的结果。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/9.png?raw=true)若考虑一个扩散模型，其中前向步长被建模为xi ∼ N (aix0, b2iI)，则Tweedie’s formula可以改写为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/10.png?raw=true)Tweedie’s formula可以应用于高斯以外的任意指数噪声分布。

## 3. 使用流行约束的条件去噪

·使用无条件训练的得分函数的同时，施加额外的约束空间。由贝叶斯规则p(x|y)=p(y|x)p(x)/p(y)可得!![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/11.png?raw=true)因此，将式(7)中反向SDE中的得分函数替换为式(11)：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/12.png?raw=true)α和W依赖于噪声协方差。

·为xi定义了集合约束，称为流行约束梯度(MCG)，以便将测量项的梯度保持在流形上。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/13.png?raw=true)

·附加流行约束下的离散反向扩散和数据一致性可以表示为：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/15.png?raw=true)

## 4. 扩散模型的几何和流形约束梯度

·符号表示：对于一个标量a，点x,y和一个集合A，定义一些符号如下：  
aA := {ax : x ∈ A};  
d(x, A) := infy∈A ||x − y||2;  
Br(A) := {x : d(x, A) < r};  
TxM: 流形M在x处的切空间；  
Jf：向量值函数f的Jacobian矩阵  
定义p0 = pdata

·假设1（强流形假设：线性结构）：假设M∈Rn是所有数据点的集合，称为数据流形。流形与维数为l << n的切空间重合。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/16.png?raw=true)
此外，数据分布p0是数据流形M上的均匀分布。传统的流形假设是关于具有低维性质的数据点的内在几何形状。在这项工作中假设更多：流形是局部线性的。

·命题1（噪声数据的集中）：考虑噪声数据的分布pi(xi) = ∫ p(xi|x)p0(x)dx, p(xi|x) ∼ N (aix, b2i I)，因此pi(xi)是集中在n-1维的流形Mi ![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/17.png?raw=true)

·扩散过程的几何解释：考虑命题1，有噪声数据的流形可以解释为两者之间的插值流形：**纯噪声N (a∞x0, b2∞)集中的超球面和干净数据流形**。在这方面，扩散步骤仅仅是从一个流形向另一个流形的过渡，扩散过程是通过插值流形从数据流形到超球体的传输。

·从命题中可以判断，只有当数据点集中在噪声数据流形上时，才能训练得分函数。因此，在远离噪声数据流形的点上应用得分函数可能会导致不准确的判断。

·命题2（分数函数）：假设sθ是式3中的去噪得分匹配损失的最小值。令Qi是对于每个i将xi映射到x0的函数。![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/18.png?raw=true)
则Qi(xi) ∈ M，J2Qi = JQi = JTQi : Rd → TQi(xi)M直觉上，Qi是M上的局部正交投影。

·根据命题2，得分函数只关注数据流形的法线方向，得分函数无法区分与流形相切的两个数据点。然而，在求解反问题时，我们希望通过区分数据点来重构原始信号，而这种区分是通过测量保真度来实现的，测量起到修正数据流形附近的切线分量的作用。

·定理1（流形约束梯度）：流形约束梯度的校正离不开数据流形，形式上![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/19.png?raw=true)梯度是数据保真项在Tx0M上的投影。

·这个定理表明，在扩散模型中，测量保真步骤将推理路径推到流形之外，并可能导致不准确的重建。另一方面，我们的流形约束修正项引导扩散位于数据流形上，从而得到更好的重建效果。几何视图如下图b所示：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/20.png?raw=true)

·可能会担心去噪分数匹配损失优化的次优性可能导致MCG步骤的不准确推断。但是，在实际中，去噪分数匹配中的大部分错误是集中在t~1上，在这样的区域，Tweedie’s inference无法生成有意义的图像，也就是说，score函数不能检测数据流形。在这个机制中，当去噪分数不准确时，MCG的幅度很小，因此由次优性引起的问题是最小的。当t→0时，估计变得精确，从而能使MCG精确实现。

## 5. Experiments

·“For experiments with CT, we train our model based on ncsnpp as a VE-SDE from score-SDE [41], on the 2016 American Association of Physicists in Medicine (AAPM) grand challenge dataset, and we process the data as in [23]. Specifically, the dataset contains 3839 training images resized to 256×256 resolution. We simulate the CT measurement process with parallel beam geometry with evenly-spaced 180 degrees. Evaluation is performed on 421 held-out validation images from the AAPM challenge.” (Chung 等, 2022, p. 7)在CT重建方面，使用AAPM数据集训练模型将基于ncsnpp的模型从score-SDE训练成VE-SDE。

·“[40] is the only method that tackles CT reconstruction directly with diffusion models. We compare our method against [40], which we refer to as score-CT henceforth. We also compare with the best-in-class supervised learning methods, cGAN [15] and SIN-4c-PRN [50]. As a compressed sensing baseline, FISTA-TV [3] was included, along with the analytical reconstruction method, FBP” (Chung 等, 2022, p. 8)CT重建的方法对照：①score - CT()；②同类最好的监督学习方法cGAN和SIN-4c-PRN进行了比较。将FISTA - TV [ 3 ]作为压缩感知基线，结合解析重构方法FBP。

·CT重建评价指标：峰值信噪比( PSNR )和SSIM进行定量评估。

·稀疏视图CT重建实验结果：![](https://github.com/wuyyao/Reading-Notes-Obsidian-/blob/main/Images/images-chungImprovingDiffusionModels2022/21.png?raw=true)

## 6. Conclusion

·MCG防止数据生成过程从流形上脱落，从而减少每一步可能累积的错误。MCG控制与数据流形相切的方向，而得分函数控制正常的方向，两个分量相互补充。

·局限性：因为扩散模型是算法的主要工作，所以所提出的方法本质上是随机的；当维度较低时，该方法有时无法产生高质量的重建；该方法采样速度慢，继承了现有扩散模型的缺点，严重依赖底层扩散模型。