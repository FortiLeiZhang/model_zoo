Faster RCNN 的 PyTorch 实现
---
如何在一张有一只猫和一条狗，以绿草地为背景的图片里，将猫和狗分别用不同颜色的框圈出来？自然而然的想到两个步骤：首先要把猫和狗 (前景) 同草地 (背景) 区分开来；第二步就是识别出哪个是猫，哪个是狗。第一步是物体的 detection 和 localization，第二步是物体的 classification。更进一步，在做 detection 的时候，首先要大体上判定图片中的哪一部分像素可能会是我们感兴趣的前景，比如图片中一片绿油油的像素点，很可能就是背景，我们就可以丢弃掉不考虑这一片区域；而图片中一半绿一半黑，两者有着一条清晰的分界线，这可能就是我们感兴趣的前景，专业称呼为 proposal。我们将图片中所有的 proposal 找出来，然后给后面的 NN 做 regression，来寻找更加精确的框坐标，这就是物体的 localization。再将框出来的部分送到 classifier 做分类识别，这就是物体的 classification。以前的方法是将找 proposal 和 classification 分开来做，训练的时候，先将图片中所有的 proposal 都找出来，然后交 classifier，前者比后者计算量要大很多，需要的时间也长。当然，这在训练的时候是没有问题的，多久都可以，但是真正用到实践当中，找 proposal 和 classification 显然只能串行计算，所以满足不了 "实时" 的要求。

Faster RCNN 的优点就在于，摒弃了所有费时费力的 proposal 算法，将 找 proposal 和 classification 都集合到一个 NN 中，而且两者共享最费时费力部分的参数。它实际上就是将一张 VGG16 网络一分为二，然后在中间插入 RPN 来产生 proposal。下面就详细研究一下 Faster RCNN，具体步骤是：从一张原始的图片开始，将它送入 Faster RCNN 中，看它究竟经历了哪些操作，最后得到了一组 bbox 和 score。

这里有个思考，facenet 作人脸识别的时候，可不可以用类似 faster rcnn 的结构？从图片中框出人脸的 mtcnn 就相当于一个 rpn，人脸识别的过程就是 classifier。

![Faster RCNN 结构图](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/FasterRCNN/doc/faster_rcnn.png)

### 1. 图片预处理
就是图中 P × Q 的图片变成 M × N 的过程。文中说 Faster RCNN 可以 'takes an image (of any size) as input'，所以这里对输入图片的大小没有限制 (实际上太小也不好使)。然后要对图片 resize 一次，文中的要求只是使最短边要大于 600，但是[实际代码中](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py)还要求最长边要小于 1000。
```python
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000
```
下文举例的时候，我们以一张 500 × 353 的图片为例，resize 以后的大小是 850 × 600。具体[图片 resize 的代码](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py#L31)是用的 cv2.resize。

### 2. 特征提取层
```python
import torch as t
from torch import nn
from torchvision.models import vgg16

def build_vgg16():
    model = vgg16()
    model.load_state_dict(t.load('/home/lzhang/model_zoo/PyTorch/FasterRCNN/pretrained/vgg16-397923af.pth'))

    features = list(model.features)[:30]
    classifier = list(model.classifier)

    del classifier[6]
    del classifier[5]
    del classifier[2]

    for layer in features[:10]:
        for p in layer.parameters():
            p.require_grad = False

    return nn.Sequential(*features), nn.Sequential(*classifier)
```
原文中分别用了 ZF Net 和 VGG16 作为基础，我们这里仅以 VGG16 为例。torchvision 里有现成的 pretrained VGG16 模型，这里选择手动加载模型参数。特征提取层实际上是 VGG16 的 0 - 29 层，包括：13 个 conv 层，13 个 relu 层，和 4 个 pooling 层。将原始的 VGG16 模型打印出来：
```python
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```
可以看到，VGG16 模型很规整：conv 层都是 (3 × 3，stride = 1，padding = 1)，这样的配置可以保证 output 和 input 的 size 是相等的；pooling 层都是 (2 × 2，stride = 2)，这样每经过一个 pooling 层，output size 变为 input size 的一半。注意，这里仅取了 0 - 29 层，第 30 层的 pooling 并没有用到，所以特征提取层只有 4 个 pooling 层，原始的图片经过特征提取层后，大小变为原来的 1 / 16。原始的 VGG16 输入 224 × 224 大小的图片，到这里就缩小成了 512 × 14 × 14 的 feature map；850 × 600 的图片就缩小成了 512 × 53 × 37 的 feature map。 这个 feature map 是下面几层的基础，所有的操作，不论是 proposal 的提取还是分类，都是针对此 feature map 的。这样做就使得两者可以共享一个 feature map，从而大大简化了计算量和计算时间。

再来看 VGG16 的 classifier。
```python
Sequential(
  (0): Linear(in_features=512 * 7 * 7, out_features=4096, bias=True)
  (1): ReLU(inplace)
  (2): Dropout(p=0.5)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)
```
classifier 的输入要求是 512 × 7 × 7 的，而我们对于输入图片的大小没有限制，所以特征提取层的输出是不规整的。所以将第 30 层的 pooling 层拿掉，替换为 RPN 和 ROI Pooling 层，使得任意形状的特征提取层的输出，经过这两层，变为形如 (512, 7 , 7)。另外，我们这里不用 dropout，最后一层 FC 的输出不再是 1000 类，所以将 classifier 的第 2 、4、6 层删除。为了进一步减小计算量，原文中还固定了 VGG16 第 0 - 9 层的参数，不再对其进行参数更新。

RPN 和 ROI Pooling 层的具体实现下文再讲。这里的思考是：这个 RPN 层太单薄了，可不可以考虑用类似 mtcnn 的结构来改造 RPN ？

### Anchor 的生成和处理
其实 anchor 的生成基本上是完全独立于 faster rcnn 的。说 “基本上是完全独立” 是因为，anchor 的生成只依赖于原始图片和 faster rcnn 特征提取层的结构，和 feature map，网络的参数没有任何决定性的关系。鉴于此，我们把 anchor 的生成放到图片的预处理过程中，如果是训练的话，还要生成 anchor regression target。不过这样做要注意 anchor 的形状，因为 VGG16 是通过 4 个 2 × 2 的 pooling layer 来进行图片尺寸减半的，所以 anchor 的大小应该是 h // 2 // 2 // 2 // 2，而不是直接 h / 16。

#### Anchor 的生成
输入 850 × 600 的图片经过 VGG16 后得到的 feature map 为原图片的 1 / 16，即 53 × 37。由于 VGG16 输入/输出的 stride 为 16，所以 feature map 上的一个点，对应到原图就是一个 16 × 16 大小的正方形区域。这个正方形区域的中心，就是所谓的 anchor。所以 feature map 上这 53 × 37 = 1961 个点，就对应着原图上 1961 个 anchor。这样我们就将原图格式化成了 1961 个以 anchor 为中心，16 × 16 大小的 base anchor box。然后对这些 base anchor box 分别施以 [8, 16, 32] 三种 scale 变形和 [0.5, 1, 2] 三种 ratio 变形，得到所有的 1961 × 9 个 anchor box。这些 anchor box 就是下文中 regression 的基础：不论是 predict bbox 的偏置还是 ground-truth bbox 的偏置，都是相对于某一个 anchor box 而言的。为了简化，将 anchor box 简写成 anchor。看透了这一点，anchor 的生成何必放在 feature map 的生成之后呢？两者完全可以并行的计算出来。具体的实现就是代码中的方法。
```python
self.meta_anchor = np.array([1, 1, self.feat_stride, self.feat_stride]) - 1
ratio_anchors = _ratio_enum(self.meta_anchor, self.ratios)
self.base_anchor = np.vstack([_scale_enum(ratio_anchors[i, :], self.scales) for i in range(ratio_anchors.shape[0])])  + 1
```
首先产生 meta_anchor，即一个 16 × 16 矩形框。然后施以三种 scale 和三种 ratio 变形，得到 9 个 base_anchor。注意，这里最后一步有个 +1 项，是为了和[原作者的实现](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py)保持一致。然后上下左右平移 16 个像素即得到所有的 anchors。这里 anchor 保存的是它在原图上的左上和右下的坐标 [x_min, y_min, x_max, y_max]。注意，VOC dataset 中的坐标存储方式为 [y_min, x_min, y_max, x_max]，有的实现方法中也使用此顺序来保存 anchor 的坐标，在代码实现时一定要确定两者是一致的。

另外，这里按照代码中的实现，得到的 anchor 是 54 × 38 = 2052 个，因为后面超出边界的 anchor 总归要去掉，所以长宽都多一个点无所谓。现在我们已经将图片格式化为 2052 × 9 = 18468 个不同中心点，不同 scale，不同 ratio 的 anchor 了，下一步的目标就是寻找一组偏置，将其施加于某个或者某些 anchor 上，从而得到一组矩形框，使得该矩形框与 ground-truth 中的 bbox 矩形框越接近越好。这就蜕化成了一个 regression 问题：即寻找一组偏置，使其与 ground truth 相对于 anchor 的偏置尽量接近。所以下面就需要找出 ground-truth bbox 相对于 anchor 的偏置。

#### Anchor Regression Target 的生成
首先要给每个 anchor 标 label，看它属于前景还是背景。分四步：
1. 删除超出图片边界的 anchor，18468 个 anchor 剩余 6372 个，
2. 对于每一个 bbox，与它 IoU 最大的 anchor 置 label 为 1，即前景。对于举例的这张图片，有 2 个 bbox，与它 IoU 最大的 anchor 共有 3 个。
3. 对于每一个 anchor，如果它与某一 bbox 的 IoU 大于 0.7，置 label 为 1。这里找到 6 个。注意，2 和 3 两个条件不是互斥的，总归找到 7 个前景 anchor。
4. 对于每一个 anchor，如果它与所有的 bbox 的 IoU 都小于 0.3，置 label 为 0，即背景。共有 5720 个。
5. 剩余的一律置 label 为 -1。共有 645 个。

从上述 label 为 0 和 1 的 anchor 中各随机选取 128 个，如果前景不够的话，用背景补足。所以，这里我们取了 7 个前景，249 个背景。

然后要计算这 256 个前景背景 anchor 相对于 ground-truth bbox 的 regression 偏置。公式在文中已经给出：
$$
\begin{aligned}
&t_x^* = (x^* - x_a) / w_a , \quad  t_y^* = (y^* - y_a) / h_a \newline
&t_w^* = \log(w^* / w_a),  \quad t_h^* = \log(h^* / h_a)
\end{aligned}
$$
这里 $(x^*, y^*, w^*, h^*)$ 是 ground-truth bbox 的中心点坐标和长宽；$(x_a, y_a, w_a, h_a)$ 是 anchor 的中心点坐标和长宽。得到的 $(t_x^*, t_y^*, t_w^*, t_h^*)$ 就是我们要拟合的 target。也就是说，对每一个被选中的前景或者背景 anchor，我们要产生一组 $(t_x, t_y, t_w, t_h)$，使得这一组参数与相对应的目标 $(t_x^*, t_y^*, t_w^*, t_h^*)$ 要尽可能接近。

最后，要将选中的 256 个 (label, reg_target) 扩充至所有的 18468 个 anchor。label 项填 -1，reg_target 项填 0。

### RPN: Region Proposal Network
上述的 anchor regression target 的生成，实际是产生了我们要拟合的目标，并没有牵扯到学习参数。从 RPN 开始才真正开始训练学习参数。
RPN 层非常简单，只有一层 3 × 3 conv 和一层 1 × 1 conv。我感觉这一层有些过于简单了，可以在这一层做些文章来提高网络的性能。首先用 512 个 3 × 3，s=1，p=1 conv 和 ReLU 将 512 × 53 × 37 的 feature map 过一下，得到的新的 feature map 和原来的是相同的。然后新的 feature map 分别经过两个支路。一个支路是 18 个 1 × 1 conv，得到 18 × 53 × 37 的结果，这是判定所有 anchor 属于前景还是背景的数值 rpn_cls_score。因为对每一个 base anchor 我们进行了 9 种变形，每一个 anchor 需要 2 个数值来判定它是前景还是背景，所以对 feature map 上的每一个点，需要 18 个数字，总共有 2 × 9 × 53 × 37 个数值，即 rpn_loc_reg。将这些数值按照第一列进行 softmax，就是每一个 anchor 是前景还是背景的概率。另一个支路是 36 个 1 × 1 conv，得到的结果是 4 × 9 × 53 × 37，不难看出，这是相对于每一个 anchor 进行 reg 所需要的数据。

下一步的工作就是在所有的 anchor 中，依据 rpn_cls_score 和 rpn_loc_reg 来选取出有效的 anchor，即所谓的 roi。具体步骤如下：

1. 根据 anchor 的坐标，将 rpn_loc_reg 的数据映射为 roi 在原图的坐标。

2. 去掉太小的 roi。这里有两点疑问：第一，原作中 min_size 取的是 16，这个好理解，因为 feature map 上一个点对应到原图中就是一片 16 × 16 的区域，但是原作中还要乘以从输入图片到原图的 scale，即从 500 × 353 到 850 × 600 的那个 scale。这里我就不理解了，所有的操作都是针对 850 × 600 的这张图，而且坐标也是针对于这张图的，这里乘以这个 scale 没有道理，所以我把它去掉了。第二，如果 min_size 取 16 的话，最小的 16 × 16 的区域得到的只是 feature map 上的一个点，在下一步进行 roi pooling 的时候，需要将剪裁出来的 feature map 抽样为 7 × 7。而对于这种情况而言，就是将一个点扩充成 7 × 7。所以这里的 min_size 取 7 × 16 = 112 是不是更合理一些。当然，这样会将图片中的一些小的目标忽略掉，但是 faster rcnn 对小物体检测结果精度偏低也是它的缺点之一，我怀疑就是因为这个原因引起的。如果取 16 的话，经过这一步，17649 个 roi 会减少到 14525 个；如果取 112 的话，只剩 9149 个。

3. 依据前景的 rpn_cls_score 对 roi 进行排序。训练时，取前 12000 个 roi，应用时，取前 6000 个。

4. 对剩余的 roi 进行 NMS。所谓的 NMS，过程是这样的：对于已经排序的 roi，选出 score 最大的一个，计算剩余的 roi 与它的 IoU，如果大于 0.7，即认为两个 roi 覆盖的是同一片区域，舍弃掉所有这样的 roi。然后重复上述步骤直至所有 roi 都被处理到。经过这一步，12000 个剩余 4230个，9149 剩余 1479 个。

5. 对于 NMS 剩余的 roi，训练时，取前 2000 个 roi，应用时，取前 300 个。

这样，就得到了所有的 roi。

### ROI Pooling
关于 roi pooling，这里有篇 [文章](https://deepsense.ai/region-of-interest-pooling-explained/) 解释的很清楚。具体过程就是：先将得到的 roi 坐标除以 16，就是 roi 上的区域在 feature map 上的区域，然后将这个区域 max pool  为 7 × 7 的，用来和下面的 classifier 对接。这里的问题就是前面我们提到的，如果 roi 太小的话，其在 feature map 上的区域小于 7 × 7，那么就不是 pooling，而是 filling 了。这里代码里用的是 nn.AdaptiveMaxPool2d，与原作的 C 实现相比效率很低。

### Classifier
classifier 就是用的 VGG16 的前两个 FC 层，将最后一个 FC 层改造成两路，一路输入判定类别的概率，并对其做 softmax；一路输出 bbox 的 reg。这里要注意的是，总的类别要加上背景这一类。由于 roi 有很多，所以最后的结果要筛选一下，先将 score 太低的去掉，再做一次 NMS。

以上就是一张图片输入到 faster rcnn 所经历的全部操作，下一步看看训练是如何进行的。















end
