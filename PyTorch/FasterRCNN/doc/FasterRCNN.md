Faster RCNN 的 PyTorch 实现
---
如何在一张有一只猫和一条狗，以绿草地为背景的图片里，将猫和狗分别用不同颜色的框圈出来？自然而然的想到两个步骤：首先要把猫和狗 (前景) 同草地 (背景) 区分开来；第二步就是识别出哪个是猫，哪个是狗。第一步是物体的 detection 和 localization，第二步是物体的 classification。更进一步，在做 detection 的时候，首先要大体上判定图片中的哪一部分像素可能会是我们感兴趣的前景，比如图片中一片绿油油的像素点，很可能就是背景，我们就可以丢弃掉不考虑这一片区域；而图片中一半绿一半黑，两者有着一条清晰的分界线，这可能就是我们感兴趣的前景，专业称呼为 proposal。我们将图片中所有的 proposal 找出来，然后给后面的 NN 做 regression，来寻找更加精确的框坐标，这就是物体的 localization。再将框出来的部分送到 classifier 做分类识别，这就是物体的 classification。以前的方法是将找 proposal 和 classification 分开来做，训练的时候，先将图片中所有的 proposal 都找出来，然后交 classifier，前者比后者计算量要大很多，需要的时间也长。当然，这在训练的时候是没有问题的，多久都可以，但是真正用到实践当中，找 proposal 和 classification 显然只能串行计算，所以满足不了 "实时" 的要求。

Faster RCNN 的优点就在于，摒弃了所有费时费力的 proposal 算法，将 找 proposal 和 classification 都集合到一个 NN 中，而且两者共享最费时费力部分的参数。它实际上就是将一张 VGG16 网络一分为二，然后在中间插入 RPN 来产生 proposal。下面就详细研究一下 Faster RCNN，具体步骤是：从一张原始的图片开始，将它送入 Faster RCNN 中，看它究竟经历了哪些操作，最后得到了一组 bbox 和 score。

这里有个思考，facenet 作人脸识别的时候，可不可以用类似 faster rcnn 的结构？从图片中框出人脸的 mtcnn 就相当于一个 rpn，人脸识别的过程就是 classifier。

![Faster RCNN 结构图]()

### 1. 图片预处理
就是图中 P × Q 的图片变成 M × N 的过程。文中说 Faster RCNN 可以 'takes an image (of any size) as input'，所以这里对输入图片的大小没有限制 (实际上太小也不好使)。然后要对图片 resize 一次，文中的要求只是使最短边要大于 600，但是[实际代码中](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py)还要求最长边要小于 1000。
```python
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000
```
下文举例的时候，就以 600 × 1000 的图片为例。具体[图片 resize 的代码](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py#L31)是用的 cv2.resize。































end
