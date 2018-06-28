ResNet 的文章有两篇：[Deep Residual Learning for Image Recognition](https://github.com/FortiLeiZhang/model_zoo/blob/master/PyTorch/ResNet/Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf) 和 [Identity Mappings in Deep Residual Networks](https://github.com/FortiLeiZhang/model_zoo/blob/master/PyTorch/ResNet/Identity%20Mappings%20in%20Deep%20Residual%20Networks.pdf)。这里仅仅关注三个问题：ResNet 解决了机器学习中的什么问题；ResNet 的代码怎么写的；ResNet 怎么用。至于文章中的性能研究和试验结果，就不做讨论了。

Deep Residual Learning for Image Recognition
---
ResNet 解决了 deep NN 的两大问题：1. deep NN 的梯度弥散和爆炸问题；2. deep NN 的精度随着模型的加深，会逐渐饱和不再上升，甚至会大幅度下降。

其理论基础在于：机器学习的目的是用一个任意复杂的函数 $\textit{H} (x)$ 来近似样本数据的分布。而这个任意复杂的函数 $\textit{H} (x)$ 可以由任意多的非线性单元来近似。如果将 $\textit{H} (x)$ 写为：$\textit{H} (x) = \textit{F} (x) + x$，形式，那么 $\textit{F} (x) = \textit{H} (x) - x$ 也可以由任意多的非线性单元来近似。而优化 $\textit{F} (x)$ 比优化 $\textit{H} (x)$ 更容易解决梯度弥散/爆炸，和性能饱和这两个问题。
注意，这最后一句的结论只是一个假设 (hypothesis)，并没有经过严格的数学证明，但是根据此设计的 ResNet 性能非常好。大概的原因是因为 shortcut 可以很好的 backprop 梯度。

### 网络结构
ResNet 通过堆叠不同数目的模块，实现了 18/34/50/101/152 五种不同深度的网络结构。其中 ResNet 18/34 采用 Basic Block 作为基本单元，而 ResNet 50/101/152 则采用 Bottlenet Block 作为基本单元。下面通过代码来深入研究一下不同深度 ResNet 是如何实现的。

#### ResNet 18/34
ResNet 18/34 由 root block，stack 1-4 组成，每一个 stack 都由 Basic Block 叠加而成，所有 Basic Block 都采用 3×3 filter。其中，stack 1 每一层有64个 filter, stack 2 每一层有128个 filter，stack 3 每一层有256个 filter，stack 4 每一层有512个 filter。stack 1-4的数目，ResNet 18 为[2, 2, 2, 2]，ResNet 34 为[3, 4, 6, 3]。

##### root block
```python
self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = nn.BatchNorm2d(num_features=64)
self.relu = nn.ReLU(inplace=True)
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```
![root block](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/root_block.jpg)

root block 这一层是对输入进行初步的处理，注意这里使用的 7×7 filter, stride=2, padding=3 与输入并不匹配，实际上最右侧的一列会被忽略掉。
$$
(224 - 7 + 2 * 3) / 2 + 1 = 112
$$
下一步的 MaxPool 同样与输入不匹配
$$
(112 - 3 + 2 * 1) / 2 + 1 = 56
$$

##### Stack 1
![stack 1](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack1.jpg)

stack 1 的 feature map 大小为56×56，与 root block 相同，所以 filter 数目也不用加倍，每一层都有64个3×3 filter。

##### Stack 2
![stack 2](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack2_1.jpg)

stack 2 的 feature map 大小为28×28，比stack 1减小一半，所以 filter 数目也要加倍，每一层有128个3×3 filter。因此，stack 2 与 stack 1 衔接的那一层要用 stride=2 的filter，同样，shortcut 也要用128个 stride=2 的 1×1 filter 使得相加的时候维度相同。
除此之外，其余层都使用如下相同结构

![stack 2](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack2_2.jpg)

##### Stack 3
stack 3 的 feature map 大小为14×14，每一层有256个3×3 filter。

##### Stack 4
stack 4 的 feature map 大小为7×7，每一层有512个3×3 filter。

##### AvgPool
采用 7×7 的 average pool，得到 (1, 512) 向量。

##### FC
采用 (512, 1000) 的 FC层，得到 (1, 1000) 向量。


#### ResNet 50/101/152
ResNet 50/101/152 由 root block，stack 1-4 组成，每一个 stack 都由 Bottleneck Block 叠加而成，所有 Bottleneck Block 都采用 1×1 filter + 3×3 filter + 1×1 filter 的组合方式来减少参数和计算量。stack 1-4的数目，ResNet 50 为[3, 4, 6, 3]，ResNet 101 为[3, 4, 23, 3]，ResNet 152 为[3, 8, 36, 3]。

##### root block
与 ResNet 18/34 相同。

##### stack 1
![stack 1](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack3.jpg)

stack 1 的 feature map 大小为56×56，与 root block 相同，所以使用 stride=1 的3×3 filter。但是filter的数目要增加到256，所以与 root block 衔接的那一层的 shortcut 也要用256个1×1 filter 来匹配。
除此之外，其余层都使用如下相同结构：[1×1, 64] + [3×3, 64] + [1×1, 256]

![stack 1](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack3_2.jpg)

##### stack 2
![stack 2](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack4.jpg)

stack 2 的 feature map 大小为28×28，比stack 1减小一半，所以使用 stride=2 的3×3 filter。但是filter的数目要增加到512，所以与 root block 衔接的那一层的 shortcut 也要用512个 stride=2 的 1×1 filter 来匹配。
除此之外，其余层都使用如下相同结构：[1×1, 128] + [3×3, 128] + [1×1, 512]

![stack 2](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/stack4_2.jpg)

##### stack 3
stack 3 的 feature map 大小为14×14，每一层结构为：[1×1, 256] + [3×3, 256] + [1×1, 1024]。

##### stack 4
stack 3 的 feature map 大小为7×7，每一层结构为：[1×1, 512] + [3×3, 512] + [1×1, 2048]。

##### AvgPool
采用 7×7 的 average pool，得到 (1, 2048) 向量。

##### FC
采用 (2048, 1000) 的 FC层，得到 (1, 1000) 向量。


### 参数初始化
```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
```
这里 conv 的 weight 用 kaiming_normal 来初始化，所有 conv 都没有 bias 项；BN 的 weight 初始化为1，bias 初始化为0。

### 输入图片的预处理
训练模型建立好了，下面看看输入图片在 feed 进 CNN 之前需要进行哪些预处理。文中是做了这些处理的：
> The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation. A (224, 224) crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted. The
standard color augmentation is used.

这里以 pytorch 中针对 imagenet 给出的[代码](https://github.com/FortiLeiZhang/model_zoo/blob/master/PyTorch/ResNet/resnet_imagenet.py)与文中所采用的方法有些不同，以 pytorch 代码中的方法为例。
```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
normalize 是对所有输入图片进行减均值除方差。均值和方差的数值是通过整个 imagenet 中的图片计算出来的，这里就当作已知的常数。

从后面的代码可以看到，输入图片的预处理方式对于 train 和 val 是不同的。下面逐行分析一下代码。

#### train dataset
```python
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
```
第一步，先进行  [RandomResizedCrop](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop)
```python
def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
    self.size = (size, size)
    self.interpolation = interpolation
    self.scale = scale
    self.ratio = ratio
```
这里除了 size 设为224，其余都用的是 default 值。
```python
def get_params(img, scale, ratio):
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            i = random.randint(0, img.size[1] - h)
            j = random.randint(0, img.size[0] - w)
            return i, j, h, w

    # Fallback
    w = min(img.size[0], img.size[1])
    i = (img.size[1] - w) // 2
    j = (img.size[0] - w) // 2
    return i, j, w, w
```
这里对原图片进行最多10次随机切割，首先计算原图片面积，然后将原面积乘以一个随机的 scale，这里的 scale 是在 (0.08, 1) 之间随机产生，也就是说，输入 CNN 的图片仅仅是原图的一部分，最极端的情况下，只有 8% 的图片会被送入 CNN 中，但是图片的 label 是不变的，所以说最坏情况下，只能用图片 8% 的信息来进行学习和分类。

然后在 (0.75, 1.3333333333333333) 之间随机取一个长宽比，计算 weight 和 height。

这里还有一步依 50% 的概率随机对换 weight 和 height。所以 weight 和 height 哪个大哪个小是随机的。

然后随机选择要切割图片的左上起始坐标 (i, j)。

如果尝试10次都不成功，那么就只能从中间开始切割。

到这里就得到了要从原图片中切割出的图片的坐标，这里切割出图片的 weight 和 height 与想要得到的图片的 size 没有任何的关系。

然后把图片 crop 出来，最后把 crop 出来的图片 resize 到想要的 size。这里要求的 size 是(224, 224)，所以如果 crop 出来的图片 weight/height 大于 224，那么就要用 interpolation 进行 downsample；反之，进行 upsample。

因此，输入 CNN 的图片与原图片完全是两码事，首先是小，其次是失真。

第二步，进行 [RandomHorizontalFlip](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip)，
就是沿着水平方向随机对切割出的图片进行翻转。然后再减均值除方差。这才得到真正 feed 进 CNN 的图片数据。

#### val dataset
```python
val_dataset = datasets.ImageFolder(valdir,
    transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
```
对于 val dataset，第一步是将原图片 resize 到256，注意，这里的256是短边的大小，resize 后的图片依旧保持原来的长宽比。然后再从中间剪裁出一个 (224, 224) 的图片进行减均值除方差后送入 CNN。

### standard 10-crop testing
文中还提到了 [10-crop testing](https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.TenCrop)
> In testing, for comparison studies we adopt the standard 10-crop testing

这里所谓的 10-crop 是指在 test 的时候，从原始图片及翻转后的图片中，从四个 corner 和 一个 center 各 crop 一个 (224, 224) 的图片，总共 10 张，然后对这10张图片进行 classification，对10次预测的结果 average。

Identity Mappings in Deep Residual Networks
---
这篇文章更进一步的讨论了网络的设计，结论有两个：
1. 不要在 shortcut 路径上做任何的操作，让输入 x 直接传到 addition 就是最好的方案。

2. 重新设计了 block，把每一个 block 中的 BN 和 ReLU 从 conv 后提到了 conv 之前，即
![resnet_pro](https://github.com/FortiLeiZhang/model_zoo/raw/master/PyTorch/ResNet/resnet_pro.jpg)
