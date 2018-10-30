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
- 这里介绍了一些通用性的网络设计原则,At this point, the utility of the principles below are speculative and additional future experimental evidence will be necessary to assess their domain of validity.  Still, grave deviations from these principles tended to result in deterioration in the quality of the networks and fixing situations where those deviations were detected resulted in improved architectures.
- 第一条准则强调了我们不能在一个bottleneck中大幅度压缩feature map的尺寸,而是应该在整个网络中从输入到输出缓慢减小.不能仅仅通过dimensionality表示信息,因为它没有考虑到结构等重要元素,the dimensionality merely provides a rough estimate of information content.
- Higher dimensional representations are easier to process locally within a network.这句话我的理解是:高维度信息更容易处理局部的信息,就是说维度更高对局部信息的表达就更好,或者说比较局部的信息用更高维度的表达更好.更多的激活函数可以更好的解耦信息,这条很重要,他能解释为什么更深层的网络为什么比更宽大(单层filter更多)的网络更好,并且网络可以训练的更快.
- 第三条强调了Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. 就是说他们认为在网络早期对空间信息进行聚合对网络表达能力没什么影响,就是说在网络前期进行对feature map形状的压缩不会对网络精度造成什么影响.
比如说在使用更多的3x3卷积之前降低feature map形状不会造成什么影响.他们猜测这个原因是相邻单元之间具有强相关性,这样在空间聚合的时候信息损失会小很多.鉴于这些信息易于压缩,压缩之后会让网络更快的学习.
- 第四条是平衡网络的深度和宽度.Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network.就是说网络可以通过调整深度和宽度达到最佳的状态.Increasing both the width and the depth of the network can contribute to higher quality networks.同时增加网路的深度和宽度可以提高网络的质量.However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel.就是说平衡增加网络深度和宽度只能线性增加网络的质量,这样效率很低.不能仅仅考虑网络质量而不考虑计算量,因为因算量是一个很大的代价,所以要平衡网络的深度和宽度,使网络达到一个最佳的状态.(这段就是说开始增加网络的深度和宽度得益于结构的设计这部分收益是非线性增加的,但是无闹加深加宽后面的收益就很少了)
- 虽然这些原则可能是有意义的，但并不是开箱即用的直接使用它们来提高网络质量。我们的想法是仅在不明确的情况下才明智地使用它们。(话都让他们说了)
## Factorizing Convolutions with Large Filter ize
- 标题的意思就是把大的Filter分解成小的.
