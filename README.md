# Summary-of-InceptionV3
## InceptionNetWork
- 对许多任务而言，卷积网络是目前最新的计算机视觉解决方案的核心。从2014年开始，深度卷积网络开始变成主流，在各种基准数据集上都取得了实质性成果。
- 非常多的计算机视觉任务都开始使用深度神经网络并取得了非常好的效果.包括目标检测[5]，分割[12]，行人姿势评估[22]，视频分类[8]，目标跟踪[23]和超分辨率[3]等.
- 众所周知Google凭借InceptionNet在2014年以巨大的优势战胜VGG获得ILSVRC2014的冠军.在当时计算资源比较昂贵,所以提高模型表达能力异常重要.
- Google使用了精心设计的Inception Module很好的提升了模型的性能,同时降低了参数量.在Inception Module中使用多尺度对不同形状的物体进行适应,
同时将较大卷积核拆分并取消了FC层压缩模型的参数,取得了非常好的效果.
- 本文Google分析了InceptionNet的成功并提出了通用性指导法则帮助我们构建高性能nn.
- 同时构建除了一种更高性能的InceptionV3,在ILSVRC 2012上达到了21.2% top-1 error和5.6% top-5 error,每次推理需要50亿次加乘运算,并且使用了不到2500万的参数.在四个模型ensemble的请款下达到了3.5% top-5 error和17.3% top-1 error.
## General Design Principles
- 这里介绍了一些通用性的网络设计原则,At this point, the utility of the principles below are speculative and additional future experimental evidence will be necessary to assess their domain of validity.  Still, grave deviations from these principles tended to result in deterioration in the quality of the networks and fixing situations where those deviations were detected resulted in improved architectures.就是说这些准则是推测性的,需要实验去验证,但是严重偏离这些准则则会使结果变差.
1. 第一条准则强调了我们不能在一个bottleneck中大幅度压缩feature map的尺寸,而是应该在整个网络中从输入到输出缓慢减小.不能仅仅通过dimensionality表示信息,因为它没有考虑到结构等重要元素,the dimensionality merely provides a rough estimate of information content(这句没看懂).
2. Higher dimensional representations are easier to process locally within a network.这句话我的理解是:高维度信息更容易处理局部的信息,就是说维度更高对局部信息的表达就更好,或者说比较局部的信息用更高维度的表达更好.更多的激活函数可以更好的解耦信息,这条很重要,他能解释为什么更深层的网络为什么比更宽(单层filter更多)的网络更好,并且网络可以训练的更快.
3. 第三条强调了Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. 就是说他们认为在网络早期(filter数量还不多的时候)对空间信息进行聚合对网络表达能力没什么影响,就是说在网络前期进行对feature map形状的压缩不会对网络精度造成什么影响.比如说在使用更多的3x3卷积之前降低feature map形状不会造成什么影响.他们猜测这个原因是相邻单元之间具有强相关性,这样在空间聚合的时候信息损失会小很多.鉴于这些信息易于压缩,压缩之后会让网络更快的学习.
- **对前面三条做下总结**.其实还是蛮自然的,第一条说我们不能在某个结构中迅速下降Grid Size的形状,不过我觉得也没人这么干吧,太蠢了.第二条说在Grid Size比较小的时候使用更多的filter(dimensional/channel,都是一个东西)更好,因为更多参数经过激活曾有利于局部特征解耦,现在普遍使用的方案都是在**Grid Size下降一半的时候让filter翻倍**,第三条就是说在filter数量少的时候降低Grid Size不会对结果造成影响,现在的网络也都是这么干的,没有说在网络后期才开始下降Grid Size,当然这么做也不科学,因为会导致前面层计算量过大.
4. 第四条是平衡网络的深度和宽度.Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network.就是说网络可以通过调整深度和宽度达到最佳的状态.Increasing both the width and the depth of the network can contribute to higher quality networks.同时增加网路的深度和宽度可以提高网络的质量.However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel.就是说平衡增加网络深度和宽度只能线性增加网络的质量,这样效率很低.不能仅仅考虑网络质量而不考虑计算量,因为因算量是一个很大的代价,所以要平衡网络的深度和宽度,使网络达到一个最佳的状态.(这段就是说开始增加网络的深度和宽度得益于结构的设计这部分收益是非线性增加的,但是无脑加深加宽后面的收益就很少了(几乎没有.))
- 虽然这些原则可能是有意义的，但并不是开箱即用的直接使用它们来提高网络质量。我们的想法是仅在不明确的情况下才明智地使用它们。(话都让他们说了)
## Factorizing Convolutions with Large Filter size
- 标题的意思就是把大的Filter分解成小的.(这段意义挺大的)
- Much of the original gains of the GoogLeNet network [20] arise from a very generous use of dimension reduction, just like in the “Network in network” architecture by Lin et al [?]. 就是说GoogLeNet的收益很大部分来源于慷慨的使用降维(这部分很重要),Consider for example the case of a 1×1 convolutional layer followed by a 3 × 3 convolutional layer.**就是说先用1x1卷及降维然后在用3x3卷及这样极大的减少了计算量**,他们的解释是:在视觉神经网络中,由于离激活层近的输出具有高相关性.因此,他们期望在激活之前降维应该可以得到极为相似的表达能力.
- 这里我们探索了不同分解卷积的方式,并且及其重视计算效率,由于InceptionNet是全卷及网络,每一个权重都意味着更多的计算.因此计算量的减少同时也意味着参数量的减少.所以通过卷及分解我们可以接偶更多的参数,并且更快的训练网络.此外我们通过有效的节省内存和计算来增加filter-bank的大小,并且可以用单个计算机来训练网络[当时由于网络参数量的庞大可能需要多个计算机一起训练,所以才会提到这个.现在对于基础网络这种情况比较少了.仅仅过了三年人们思考的问题已经不是如何在一台机器上训练,而是怎么把之前训练好几周的网络一个小时训练完LOL.](https://arxiv.org/abs/1706.02677)
### Factorization into smaller convolutions. 
- 分解到更小的卷积,larger spatial filters意味着更大的计算量,比如5x5有n个filter要比相同filter的3x3卷积多25/9 = 2.78倍计算.**Of course, a 5×5 filter can capture dependencies between signals between activations of units further away in the earlier layers, so a reduction of the geometric size of the filters comes at a large cost of expressiveness.** 5x5的卷积可以在激活之前捕获到更大的面积的信号之间的关联性,因此在几何倍数的降低filters的大小会有巨大的损失.**但是我们可以把5x5卷积分解成两个连续的3x3卷积.** However, we can ask whether a 5 × 5 convolution could be replaced by a multi-layer network with less parameters with the same input size and output depth. If we zoom into the computation graph of the 5 × 5 convolution, we see that each output looks like a small fully-connected network sliding over 5×5 tiles over its input (see Figure 1). Since we are constructing a vision network, it seems natural to exploit translation invariance again and replace the fully connected component by a two layer convolutional architecture:the first layer is a 3 × 3 convolution, the second is a fully connected layer on top of the 3 × 3 output grid of the first layer (see Figure 1). Sliding this small network over the input activation grid boils down to replacing the 5 × 5 convolution with two layers of 3 × 3 convolution (compare Figure 4 with 5).如果我们放大5×5卷积的计算图，我们看到每个输出看起来像一个小的完全连接的网络，在其输入上滑过5×5的块（见图1）。由于我们正在构建视觉网络，所以通过两层的卷积结构再次利用平移不变性来代替全连接的组件似乎是很自然的：第一层是3×3卷积，第二层是在第一层的3×3输出网格之上的一个全连接层（见图1）。通过在输入激活网格上滑动这个小网络，用两层3×3卷积来替换5×5卷积（比较图4和5）。
### Spatial Factorization into Asymmetric Convolutions
- 空间分解为不对称卷积(反正没人用).上述结果表明，大于3×3的卷积滤波器可能不是通常有用的，因为它们总是可以简化为3×3卷积层序列。我们仍然可以问这个问题，是否应该把它们分解成更小的，例如2×2的卷积。然而，通过使用非对称卷积，可以做出甚至比2×2更好的效果，即n×1。例如使用3×1卷积后接一个1×3卷积，相当于以与3×3卷积相同的感受野滑动两层网络（参见图3）。如果输入和输出滤波器的数量相等，那么对于相同数量的输出滤波器，两层解决方案便宜33％。相比之下，将3×3卷积分解为两个2×2卷积表示仅节省了11％的计算量。在理论上，我们可以进一步论证，可以通过1×n卷积和后面接一个n×1卷积替换任何n×n卷积，并且随着n增长，计算成本节省显著增加（见图6）。实际上，我们发现，采用这种分解在前面的层次上不能很好地工作，但是对于中等网格尺寸（在m×m特征图上，其中m范围在12到20之间），其给出了非常好的结果。在这个水平上，通过使用1×7卷积，然后是7×1卷积可以获得非常好的结果。这段解释了如何把nxn卷积分解为1xn和nx1,这是另一种减小参数量的方法,实际运用时没有造成实验精度变低.思想很好,但是没人用.(现在也没用了,设计思路变了).
## Utility of Auxiliary Classifiers
- 利用辅助分类器:已经弃用的方法.引入了辅助分类器的概念，以改善非常深的网络的收敛。最初的动机是将有用的梯度推向较低层，使其立即有用，并通过抵抗非常深的网络中的消失梯度问题来提高训练过程中的收敛。Lee等人[11]也认为辅助分类器促进了更稳定的学习和更好的收敛。有趣的是，我们发现辅助分类器在训练早期并没有导致改善收敛：在两个模型达到高精度之前，有无侧边网络的训练进度看起来几乎相同。接近训练结束，辅助分支网络开始超越没有任何分支的网络的准确性，达到了更高的稳定水平。另外，[20]在网络的不同阶段使用了两个侧分支。移除更下面的辅助分支对网络的最终质量没有任何不利影响。再加上前一段的观察结果，这意味着[20]最初的假设，这些分支有助于演变低级特征很可能是不适当的。相反，我们认为辅助分类器起着正则化项的作用。这是由于如果侧分支是批标准化的[7]或具有丢弃层，则网络的主分类器性能更好。这也为推测批标准化作为正则化项给出了一个弱支持证据。论文提到这是一种正则话的方法,在没有ResidualBlock之前是有用的,因为不加的话前面的层容易梯度消失不好训练,但是之后就没用了.
## Efficient Grid Size Reduction
- 有效的网格尺寸减少.这一段也很有参考意义.传统上，卷积网络使用一些池化操作来缩减特征图的网格大小。为了避免表示瓶颈，在应用最大池化或平均池化之前，需要扩展网络滤波器的激活维度。例如，开始有一个带有k个滤波器的d×d网格，如果我们想要达到一个带有2k个滤波器的d2×d2网格，我们首先需要用2k个滤波器计算步长为1的卷积，然后应用一个额外的池化步骤。这意味着总体计算成本由在较大的网格上使用2d2k2次运算的昂贵卷积支配。一种可能性是转换为带有卷积的池化，因此导致2(d2)2k2次运算，将计算成本降低为原来的四分之一。然而，由于表示的整体维度下降到(d2)2k，会导致表示能力较弱的网络（参见图9），这会产生一个表示瓶颈。我们建议另一种变体，其甚至进一步降低了计算成本，同时消除了表示瓶颈（见图10），而不是这样做。**我们可以使用两个平行的步长为2的块：P和C。P是一个池化层（平均池化或最大池化）的激活，两者都是步长为2，**其滤波器组连接如图10所示。这里提出一个结构如图10所示,在这个结构中用stride=2和pooling层共同压缩Grid Size(就是filter size),现在一般只使用stride=2的conv来做而不和pooling做concat.
## Inception-v3
- 基于以上的讨论提出Inception-v3网络长这个样子.

|type|patch size/stride|input size|
| :------ | :------: | :------: |
|conv| 3×3/2| 299×299×3|
|conv |3×3/1| 149×149×32|
|conv padded |3×3/1| 147×147×32|
|pool| 3×3/2 |147×147×64|
|conv| 3×3/1| 73×73×64|
|conv| 3×3/2| 71×71×80|
|conv| 3×3/1|35×35×192|
|3×Inception| As in figure 5| 35×35×288|
|5×Inception| As in figure 6 |17×17×768|
|2×Inception| As in figure 7 |8×8×1280|
|pool| 8 × 8| 8 × 8 × 2048|
|linear| logits| 1 × 1 × 2048|
|softmax| classifier| 1 × 1 × 1000|

- 注意，基于与3.1节中描述的同样想法，我们将传统的7×7卷积分解为3个3×3卷积。对于网络的Inception部分，我们在35×35处有3个传统的Inception模块，每个模块有288个滤波器。使用第5节中描述的网格缩减技术，这将缩减为17×17的网格，具有768个滤波器。这之后是图5所示的5个分解的Inception模块实例。使用图10所示的网格缩减技术，这被缩减为8×8×1280的网格。在最粗糙的8×8级别，我们有两个如图6所示的Inception模块，每个块连接的输出滤波器组的大小为2048。网络的详细结构，包括Inception模块内滤波器组的大小，在补充材料中给出，在提交的tar文件中的model.txt中给出。然而，我们已经观察到，只要遵守第2节的原则，对于各种变化网络的质量就相对稳定。虽然我们的网络深度是42层，但我们的计算成本仅比GoogLeNet高出约2.5倍，它仍比VGGNet要高效的多。

## Model Regularization via Label Smoothing
- [参考](http://cxsmarkchan.com/articles/ml-regularization-label-smoothing.html)

## Training Methodology
- 我们在TensorFlow[1]分布式机器学习系统上使用随机梯度下降法训练了我们的网络，使用了50个NVidia Kepler GPU训练，每块上batch_size=32，跑了100个epoch。我们之前的实验使用动量方法[19]，衰减值为0.9，而我们最好的模型是用RMSProp [21]实现的，衰减值为0.9，ϵ=1.0。我们使用0.045的学习率，每两个epoch以0.94的指数速率衰减。此外，阈值为2.0的梯度裁剪[14]被发现对于稳定训练是有用的。使用随时间计算的运行参数的平均值来执行模型评估。

## Performance on Lower Resolution Input
- 视觉网络的典型用例是用于检测的后期分类，例如在Multibox [4]上下文中。这包括分析在某个上下文中包含单个对象的相对较小的图像块。任务是确定图像块的中心部分是否对应某个对象，如果是，则确定该对象的类别。这个挑战的是对象往往比较小，分辨率低。这就提出了如何正确处理低分辨率输入的问题。 普遍的看法是，使用更高分辨率感受野的模型倾向于导致显著改进的识别性能。然而，区分第一层感受野分辨率增加的效果和较大的模型容量、计算量的效果是很重要的。如果我们只是改变输入的分辨率而不进一步调整模型，那么我们最终将使用计算上更便宜的模型来解决更困难的任务。当然，由于减少了计算量，这些解决方案很自然就出来了。为了做出准确的评估，模型需要分析模糊的提示，以便能够“幻化”细节。这在计算上是昂贵的。因此问题依然存在：如果计算量保持不变，更高的输入分辨率会有多少帮助。确保不断努力的一个简单方法是在较低分辨率输入的情况下减少前两层的步长，或者简单地移除网络的第一个池化层。
- 为了这个目的我们进行了以下三个实验：
  - 步长为2，大小为299×299的感受野和最大池化。
  - 步长为1，大小为151×151的感受野和最大池化。
  - 步长为1，大小为79×79的感受野和第一层之后没有池化。
- 所有三个网络具有几乎相同的计算成本。虽然第三个网络稍微便宜一些，但是池化层的成本是无足轻重的（在总成本的1\％以内）。在每种情况下，网络都进行了训练，直到收敛，并在ImageNet ILSVRC 2012分类基准数据集的验证集上衡量其质量。结果如表2所示。虽然分辨率较低的网络需要更长时间去训练，但最终结果却与较高分辨率网络的质量相当接近。

## Experimental Results and Comparisons
- 表3显示了我们提出的网络（Inception-v3）识别性能的实验结果，架构如第6节所述。每个Inception-v3行显示了累积变化的结果，包括突出显示的新修改加上所有先前修改的结果。标签平滑是指在第7节中描述的方法。分解的7×7包括将第一个7×7卷积层分解成3×3卷积层序列的改变。BN-auxiliary是指辅助分类器的全连接层也BN的版本，而不仅仅是卷积。我们将表3最后一行的模型称为Inception-v3，并在多裁剪图像和组合设置中评估其性能。我们所有的评估都在ILSVRC-2012验证集上的48238非错单样本中完成，如[16]所示。我们也对所有50000个样本进行了评估，结果在top-5错误率中大约为0.1%，在top-1错误率中大约为0.2%。在本文即将出版的版本中，我们将在测试集上验证我们的组合结果，但是我们上一次对BN-Inception的春季测试[7]表明测试集和验证集错误趋于相关性很好。
## Conclusions
- 我们提出了几个设计准则来指导大规模卷及神经网络构建,并且在Inception上做了实验.与更简单，更单一的架构相比，这些准组指导了相对适中的计算成本的高性能视觉网络。我们的高质量Inception-v2达到了21.2%, top-1 and 5.6% top-5 error for [single crop evaluation](http://www.caffecn.cn/?/question/428) on the ILSVR 2012 classification, setting a new state of the art.现在测试都是single crop evaluation,其他的结果都不用看.
- 与Ioffe等[7]中描述的网络相比，这是通过增加相对（2.5/times）的计算成本来实现的。尽管如此，我们的解决方案所使用的计算量比基于更密集网络公布的最佳结果要少得多：我们的模型比He等[6]的结果更好——将top-5(top-1)的错误率相对分别减少了25% (14%)，然而在计算代价上便宜了六倍，并且使用了至少减少了五倍的参数（估计值）。我们的四个Inception-v3模型的[ensemble](https://www.cnblogs.com/zhizhan/p/5051881.html)效果达到了3.5\％，多裁剪图像评估达到了3.5\％的top-5的错误率，这相当于比最佳发布的结果减少了25\％以上，几乎是ILSVRC 2014的冠军GoogLeNet组合错误率的一半。(然而你最终还是被He et al打败了.)
- 我们还表明，可以通过感受野分辨率为79×79的感受野取得高质量的结果。这可能证明在检测相对较小物体的系统中是有用的。我们已经研究了在神经网络中如何分解卷积和积极降维可以降低网络的计算成本，同时保持高质量。较低的参数数量、额外的正则化、批标准化的辅助分类器和标签平滑的组合允许在相对适中大小的训练集上训练高质量的网络。

## Summary
   Google出品必属良心呀.前面的指导准则都是比较有参考意义的.后面的实验做到让人眼花缭乱.可以看出之后的网络都很大的参考了其中一部分.既然说到了之后的网络那就说说ResNet吧,基于残差的学习方式加上之前的工作可以说让神经网络的训练难易度降到了最低.在没有BN没有Residual的那年代网络训练异常艰苦而复杂,当然我也没经历那年代,所以可能体会的还不够深刻,所以就不多说了.现在人们在构造复杂任务(比如检测,识别)的时候往往使用VGG或者ResNet,VGG的优点是可理解性极强,它比较符合人们对卷积和Pooling基本作用的理解,简单直接,没有任何骚操作.而ResNet的出现让人们不在钟情Inception,或许在初学者眼里两者可能都不太好理解,都很复杂,有很多的新东西新理论出现在里面,乱七八糟的,但是想比较而言ResNet还是比Inception简单很多,无论是在结构上还是在细节上.但是不可否认的是,如果对ResNet有足够了解的读者,看到这里应该很明显的感觉到ResNet,乃至更近期的网络都参考很多本文的4条理念.当然Residual的出现也让一些类似Utility of Auxiliary Classifiers这样复杂的东西退出历史舞台了.比如说ResNet开始使用filter_size=7的卷积进行降维,印证了本文提到的不能直接使用3x3卷积进行降维否则会造成空间信息的丢失,以及在网络前期filter数量相对较少时进行进行维度压缩不会造成精度损失.以及提出先用1x1卷及降维然后在用3x3卷及这样极大的减少了计算量而不会损失精度的方法.都是非常具有参考意义的(之前真没想到是这里提出来的).读完这篇论文感觉ResNet的贡献就只有设计出了skip connect.以及使用前人总结的经验理论创造出了ResNet.
