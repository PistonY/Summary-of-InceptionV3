# Summary-of-Rethinking-the-Inception-Architecture-for-Computer-Vision
## InceptionNetWork
- 众所周知Google凭借InceptionNet在2014年以巨大的优势战胜VGG获得ILSVRC2014的冠军.在当时计算资源比较昂贵,所以提高模型表达能力异常重要.
- Google使用了精心设计的Inception Module很好的提升了模型的性能,同时降低了参数量.在Inception Module中使用多尺度对不同形状的物体进行适应,
同时将较大卷积核拆分并取消了FC层压缩模型的参数,取得了非常好的效果.
- 本文Google分析了InceptionNet的成功并提出了通用性指导法则帮助我们构建高性能nn.
