MTCNN 的 TensorFlow 实现
---
Magic Vision 要加人脸识别功能，所以要在 TensorFlow Serving 上起一个人脸识别服务，自然想到的是 Google 的 [Facenet](https://github.com/davidsandberg/facenet)。 由于 Google 官方提供下载的 Facenet 模型中有个 [bug](https://github.com/davidsandberg/facenet/issues/789) 导致其 serving 不起来，所以要在他们的源代码上进行修改重新训练。于是乎我想干脆把他们的代码自己重新写一遍算了。至于如何从头开始训练 Facenet，参见这篇 [文章](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)。

Facenet 的实现过程包括两步，首先是用 MTCNN 将图片中的人脸框出来，第二步是识别框出来的人脸是谁。这里先完成第一步，即 MTCNN 的 TensorFlow 实现，并将得到的 model 在 TensorFlow Serving 上跑起来。

MTCNN [原始论文](https://arxiv.org/abs/1604.02878) 中的 [代码](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 是用 MATLAB 实现的。Facenet 只是将 MATLAB 代码翻译成了 TensorFlow , 并使用已经训练好的模型参数。这里不得不吐槽一下原始论文中的 MATLAB 代码，到处都是多余的 T 啊，有事没事的就来个转置，这肯定是平时写论文推公式养成的习惯，见到个矩阵后面就加个 T，本来好好的 (x, y) 坐标，非得要加个 T 变成 (y, x)，完事再 T 回来。Google 也耿直，翻译代码的时候也是见到 T 就 np.transpose。我试着将多余的转置去掉，发现不行，结果不对，可能现成的参数就是这么训练出来的，如果去掉转置的话，参数可能对不上。长话短说，还是先看代码吧。

### 1. Dataset : CASIA-maxpy-clean
Facenet 用的是 CASIA-webface dataset 进行训练。这个 dataset 在原始地址已经下载不到了，而且这个 dataset 据说有很多无效的图片，所以我用的是清理过的数据库。该数据库在百度网盘有下载：[下载地址](http://pan.baidu.com/s/1kUdRRJT)，提取密码为 3zbb。

这个数据库有 10575 个类别，每个类别都有各自的文件夹，里面有同一个人的几张或者几十张不等的脸部图片。MTCNN 的工作就是从这些照片中把人物的脸框出来，然后交给下面的 Facenet 去处理。这里建立一个 ImageClass，存储各个类别的编号名称和该类别下所有图片的绝对路径。

### 2. 建立 MTCNN 模型
首先要在 main 函数中起一个 Graph，模型的图就建在这个 Graph 中，然后在此 Graph 中起一个 session 来运行函数执行命令建立三个 CNN：Proposal Network (P-Net), Refine Network (R-Net) 和 Output Network (O-Net)。
```python
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
```

Google Facenet 的原作者在建立网络时，自己重写了 CNN 网络所需要的各个组件，包括 Conv 层，MaxPool 层，Softmax 层等等。这里我偷点懒，用现成的 Keras 来实现各个组件。这里先只关注网络是如何搭建的，至于网络的输入输出以及是如何运作的，在下一节细说。

#### PNet
首先建立一个 variable_scope，在此 scope 中的所有 variable 和 op 的名称都会加前缀 pnet/ 。

输入是一个形如 (None, None, None, 3) 的 placeholder。

然后就是根据文章中的参数建立模型就好了。这里需要注意的地方有两处：

###### PReLU 层
MTCNN 使用 Parametric ReLU (PReLU) 来引入 nonlinearity，PReLU 的定义如下：
$$
f(x) = \left\{\begin{matrix}
 x& \text{if} \quad x > 0\newline
 \alpha \cdot x & \text{otherwise}
\end{matrix}\right.
$$
与 Leaky ReLU 不同的是，PReLU 中的参数 $\alpha$ 是一个需要学习的参数，而 Leaky ReLU 中的 $\alpha$ 仅仅是一个预先设定好的超参数。在使用 PReLU 时，如果输入的形状是 (H, W, C) 的，那么对于同一个 channel ，这个参数 $\alpha$ 是相同的，所以要设置 shared_axes：
```python
self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2], name='PReLU1')
```

###### Softmax 层
同样的道理，对于形如 (H, W, C) 的输入做 Softmax，针对的是不同的 channel，所以：
```python
self.softmax = tf.keras.layers.Softmax(axis=3, name='prob1')
```

###### Output Tensor
这里的输出是形如 (H, W, 2) 的 face classification，和形如 (H, W, 4) 的bounding box regression，并没有输出形如 (H, W, 10) 的 facial landmark localization。至于每个输入的意义和形状的意义，下一节细讲。

###### call 函数
在实现 call 函数的时候要注意两点，首先是 PNet 有两路输出，所以在两路分叉前，要将输入变量复制一份：
```python
_x = tf.identity(x)
```
另外，call 函数实际上不需要返回值，但是不提供返回值，或者是返回 None，TensorFlow 都会报错，所以这里返回个 tf.zeros([1]) 糊弄一下，应该有更有意义的返回值，以后 TensorFlow 用熟了再回来改。

###### 参数加载
由于对 TensorFlow 不熟，所以在参数加载这里颇费了些功夫。这里把学习的过程记录下来。

Facenet 的作者提供了三个文件，其中 "det1.npy" 就是 PNet 中每一层的参数。这里要作的是，找到 "det1.npy" 中每一层的名称和数据，然后找到刚才我们建立的 CNN 中每一层的名称，将数据赋给相应的参数。由于 Facenet 的作者是自己实现的 CNN 的每一个模块，所以在实现模块时他已经把每一层的名称和 "det1.npy" 中每一层的名称一致起来了。我们使用 keras 实现的 CNN，每一层的名称和 "det1.npy" 中是不同的。

写个函数将 "det1.npy" 中的数据打印出来
```python
from six import string_types, iteritems

data_dict = np.load('/home/lzhang/model_zoo/TensorFlow/mtcnn/det1.npy', encoding='latin1').item()
for op_name in data_dict:
    for param_name, data in iteritems(data_dict[op_name]):
        print(op_name, ":", param_name, data.shape)

conv4-2 : weights (1, 1, 32, 4)
conv4-2 : biases (4,)
conv4-1 : weights (1, 1, 32, 2)
conv4-1 : biases (2,)
conv3 : weights (3, 3, 16, 32)
conv3 : biases (32,)
conv2 : weights (3, 3, 10, 16)
conv2 : biases (16,)
conv1 : weights (3, 3, 3, 10)
conv1 : biases (10,)
PReLU1 : alpha (10,)
PReLU2 : alpha (16,)
PReLU3 : alpha (32,)
```
将新建 PNet 中的变量打印出来：
```python
def debug_print_tensor_variables():
    tensor_variables = tf.global_variables()
    for variable in tensor_variables:
        print(str(variable))

<tf.Variable 'pnet/p_net/conv1/kernel:0' shape=(3, 3, 3, 10) dtype=float32>
<tf.Variable 'pnet/p_net/conv1/bias:0' shape=(10,) dtype=float32>
<tf.Variable 'pnet/p_net/PReLU1/alpha:0' shape=(1, 1, 10) dtype=float32>
<tf.Variable 'pnet/p_net/conv2/kernel:0' shape=(3, 3, 10, 16) dtype=float32>
<tf.Variable 'pnet/p_net/conv2/bias:0' shape=(16,) dtype=float32>
<tf.Variable 'pnet/p_net/PReLU2/alpha:0' shape=(1, 1, 16) dtype=float32>
<tf.Variable 'pnet/p_net/conv3/kernel:0' shape=(3, 3, 16, 32) dtype=float32>
<tf.Variable 'pnet/p_net/conv3/bias:0' shape=(32,) dtype=float32>
<tf.Variable 'pnet/p_net/PReLU3/alpha:0' shape=(1, 1, 32) dtype=float32>
<tf.Variable 'pnet/p_net/conv4-1/kernel:0' shape=(1, 1, 32, 2) dtype=float32>
<tf.Variable 'pnet/p_net/conv4-1/bias:0' shape=(2,) dtype=float32>
<tf.Variable 'pnet/p_net/conv4-2/kernel:0' shape=(1, 1, 32, 4) dtype=float32>
<tf.Variable 'pnet/p_net/conv4-2/bias:0' shape=(4,) dtype=float32>
```
这就一目了然了。需要注意的是，由于函数在调用的时候在 variable_scope('pnet') 中，所以在 tf.get_variable() 时，变量名会自动前缀一个 "pnet/"，不要重复添加了。这里还需要注意的是，alpha 的维度不同，需要将两者匹配起来。

###### 调用 PNet
PNet 建立起来了，我们需要将图片 feed 给它，得到输出 tensor。输入 tensor 我们已经定义为名为 "pnet/input" 的 placeholder。那么输出 tensor 的名字是什么呢。 定义一个函数，将 PNet 中的所有 operation 的名称打印出来：
```python
def debug_print_tensor_operations():
    with open('/home/lzhang/tensorflow_debug.txt', 'w') as f:
        for op in tf.get_default_graph().get_operations():
            f.write(str(op))
```
显然我们需要两个 tensor 的输出，一个是 "pnet/p_net/conv4-2" 层的输出，另一个是 "pnet/p_net/prob1" 层的输出。但是这两层最后一个 tensor 的名字到底是什么？我们搜索文档，看到最后一个以 "pnet/p_net/conv4-2" 为前缀的 op 名称为 "pnet/p_net/conv4-2/BiasAdd"；最后一个以 "pnet/p_net/prob1" 为前缀的 op 名称为 "pnet/p_net/prob1/truediv"。这也符合我们的预期，因为 conv4-2 最后要加上 bias 然后输出，而 softmax 最后一步肯定是 divide。所以我们想要的输出值就是这两个 tensor 的输出值。不要忘记在后面还要加一个 ":0"。如何还有别的方法来确定 tensor 的名称的话，学会以后回来再补。

所以，我们要 run 一个 session，从 Graph 中的输入 tensor "pnet/input:0", 运行到 "pnet/p_net/conv4-2/BiasAdd:0" 和 "pnet/p_net/prob1/truediv:0"，并将两个 tensor 的输出结果作为返回值，所以这个函数指针为：
```python
pnet_fun = lambda img : sess.run(('pnet/p_net/conv4-2/BiasAdd:0', 'pnet/p_net/prob1/truediv:0'), feed_dict={'pnet/input:0':img})
```

RNet 和 ONet 按照上述同样的方法搭建起来，整个 MTCNN 就搭建完成了。

### 3. MTCNN 数据处理流程
一张图片，从输入进 MTCNN，到最后将人脸框出来输出，MTCNN 究竟对这张图片进行了怎样的处理，我们一步一步来仔细研究一下 MTCNN 的数据处理流程。根据 MTCNN 中的三张网络 PNet， RNet 和 ONet，处理过程自然的分三步。

#### PNet
![PNet](https://github.com/FortiLeiZhang/model_zoo/raw/master/TensorFlow/mtcnn/PNet.jpg)

先来看 PNet 的结构。注意到 PNet 中，除了 Conv，PReLU，MaxPool 以外，并没有 FC 层，所以 PNet 最终的输出是一张 (H, W, 16) 的 feature map，而不是一个 (16, ) 的 vector。实际上，在这里的代码实现中，10维的 landmark 并没有输出，所以输出是 (H, W, 6) 的特征图。通过计算，可以得到，特征图中的每一个 1×1 的特征点，对应在原图中的视野是 12×12 的，所以这就是上图中 input size 为 12×12×3 的原因。这不表示 PNet 的输入大小是 12×12×3，实际上，输入图片最小是 12×12 的，其他并没有限制。所以对于 PNet 来说，输入一张 (h, w, c) 的图片，输出一张 (H, W, 6) 的特征图。相当于用 12×12 的 block 在原图上以 stride = 2 来滑动，输出一张特征图。 (h, w) 和 (H, W) 的关系是可以通过计算得到的，很简单。

###### 输入图片的 rescale

明白了 PNet 的结构以后，接下来看代码：
```python
minl=np.amin([h, w])
m=12.0/minsize
minl=minl*m
# create scale pyramid
scales=[]
while minl>=12:
    scales += [m*np.power(factor, factor_count)]
    minl = minl*factor
    factor_count += 1

# first stage
for scale in scales:
    hs=int(np.ceil(h*scale))
    ws=int(np.ceil(w*scale))
    im_data = imresample(img, (hs, ws))
```
PNet 能处理的最小图片大小是 12×12 的， minsize 设为20，即要求的最小人脸图片的大小是 20×20 的，根据这两个值定义了一个 detection_window_size_ratio 的值 m = 12.0 / minsize。然后定义了一组图片的 scales：
$$
\text{scale} = m \cdot \text{factor}^n
$$
factor 在这里取 0.709。输入图片按照这些 scales 进行缩放，直到最短边的值小于 12 为止。从而得到了一组内容相同，大小不同的image pyramid。注意，这里对图片进行的是 resample，并不是 crop
```python
def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data
```
不过让我奇怪的是，在 resample 这个函数里，h，w 的值是颠倒的，我查了一下 cv2.resize 的[说明](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)，居然真的是这样。也就是说，如果想得到一个 (30, 40) 的图片，在调用 cv2.resize 的时候，参数要传入 (40, 30)。为了确保万一，我写了段代码验证一下：
```python
import cv2
from scipy import misc

img = misc.imread('/home/lzhang/tmp/0000045/001.jpg')
print(img.shape)
img2 = cv2.resize(img, (30, 40), interpolation=cv2.INTER_CUBIC)
print(img2.shape)
img3 = cv2.resize(img, (40, 30), interpolation=cv2.INTER_CUBIC)
print(img3.shape)

(250, 250, 3)
(40, 30, 3)
(30, 40, 3)
```
我不得不说，cv2 这个源码的作者，您的逻辑真是太清奇了，这么搞，不知道多少人要在这里出错啊。看 Google 的原始版本的代码，里面保留了一段 debug 代码，而且原作者在这个 [issue 的回复中](https://github.com/davidsandberg/facenet/issues/49) 里也提到了这段代码。我估计原来作者就是用 debug 这段代码来实现 resample 的，后来发现效率太低，速度太慢，转而用 cv2 实现，实现的时候也遇到了这个问题，所以才把这段代码留在这里。

###### pnet 输出

继续往下走，把输入图片 resample 以后，常规操作减均值除方差，然后转置一下送到 pnet 中。这里的转置是多余的，但是因为我们是直接 load 原作者的网络模型参数来用，所以一定要完全按照他训练时处理数据的方法来一模一样的处理数据。得到的 pnet 的输出，再转置回来。

这里要检查一下输出的 out 对不对。具体方法是，输入同一张图片，用 Google 的原版代码产生一组 pnet，rnet 和 onet，在 pnet 的输出 out 后加一个返回值，自己的代码同样返回 pnet 的输出值，然后看两组 out 值差的绝对值之和，理论上应该为 0 或者很小的一个数字。下面写到 rnet 和 onet 的时候同样要作此检查。

下一步看看 pnet 的输出值的实际意义是什么。首先看 out0 和 out1 的形状：
```python
print(img.shape)
print(out0.shape)
print(out1.shape)
(250, 250, 3)
(1, 70, 70, 4)
(1, 70, 70, 2)
```
我们算一下：输入图片的大小是 250×250×3，首先要 scale，这里 minsize = 20，m = 12 / minsize = 3/5，实际输入 pnet 的第一张图片的大小是 250 × 3/5 = 150，经过第一层 (3×3/s:1/p:valid) conv，输出为 (150 - 3 + 1) / 1 = 148；第二层 (2×2/s:2/p:same) maxpool，输出为 148 / 2 = 74；第三层 (3×3/s:1/p:valid) conv，输出为 (74 -3 + 1) / 1 = 72；第四层 (3×3/s:1/p:valid) conv，输出为 (72 -3 + 1) / 1 = 70，所以输出的 feature map 应该形如 (70, 70)。

来看 out0，它的形状是 (1, 70, 70, 4)，是 boudingbox regression，我们随机打印一个出来：
```python
[-0.04171251 -0.03393787 -0.05021905  0.14131135]
```
这些值具体是坐标，还是偏置，目前还看不出来。

来看 out1，它的形状是 (1, 70, 70, 2)，是 face classification，随机打印一个：
```python
[0.9564381  0.04356182]
```
两者和是1，显然是一个概率值。第一个数字应该表示0，即不是人脸的概率，第二个数字表示是1，即是人脸的概率。

###### 生成 boundingbox
接下来是要产生 boundingbox，由于输入的图片经过了一次转置，所以接下来的所有操作都要转置来转置去，这个 [问题](https://github.com/ipazc/mtcnn/issues/4) 也有人问过原作者，据说是 MATLAB Caffe 的什么东西引起的，也没法解决。所以函数 generateBoundingBox 代码里的细节就不讨论了，只说说它的作用。首先它从输出的 feature map 中，找出判定是人脸的概率大于 threshold (这里取值 0.6) 的点的坐标，然后将这个坐标回溯出它在原图中的坐标。我们在上面讲过，特征图相当于用 12×12 的 block 在原图上以 stride = 2 来滑动得到的，特征图中的一个点的坐标，相当于原图中的一个 12×12 的 block，这个 block 的起始和终点坐标为：
```python
q1 = np.fix((bb * stride + 1) / scale)
q2 = np.fix((bb * stride + cell_size) / scale)
```
最终输出的 boundingbox 是形如 (x, 9)，其中前4位是 block 在原图中的坐标，第5位是判定为人脸的概率，后4位是 boundingbox regression 的值。具体 boundingbox regression 到底是什么，现在还不清楚。

###### NMS

NMS (Non-Maximum Suppression)：在上述生成的 bb 中，找出判定为人脸概率最大的那个 bb，计算出这个 bb 的面积，然后计算其余 bb 与这个 bb 重叠面积的大小，用重叠面积除以：(Min) 两个 bb 中面积较小者；(Union) 两个 bb 的总和面积。如果这个值大于 threshold，那么就认为这两个 bb 框的是同一个地方，舍弃判定概率小的；如果小于 threshold，则认为两个 bb 框的是不同地方，保留判定概率小的。重复上述过程直至所有 bb 都遍历完成。

将图片按照所有的 scale 处理过一遍后，会得到在原图上基于不同 scale 的所有的 bb，然后对这些 bb 再进行一次 NMS，并且这次 NMS 的 threshold 要提高。

###### 校准 bb
从这一步可以看出 bb regression 到底表示什么意义了。

```python
reg_w = total_boxes[:, 2] - total_boxes[:, 0]
reg_h = total_boxes[:, 3] - total_boxes[:, 1]
qq1 = total_boxes[:, 0] + total_boxes[:, 5] * reg_w
qq2 = total_boxes[:, 1] + total_boxes[:, 6] * reg_h
qq3 = total_boxes[:, 2] + total_boxes[:, 7] * reg_w
qq4 = total_boxes[:, 3] + total_boxes[:, 8] * reg_h
```
显然，bb regression 是基于长宽 (h, w) 的相对于坐标 (x, y) 的偏置。原始的坐标 (x, y) 加上偏置以后，就得到了 pnet 校准后的 bb 坐标。接着还要把框调整一下成为一个正方形。最后一步是把超过原图边界的坐标剪裁一下。这就得到了真真正正的在原图上 bb 的坐标。

到此为止，第一步 PNet 的任务就完成了，下一步工作交给 RNet。

#### RNet

RNet 的输入是 PNet 产生的所有 bb。不论 bb 的实际的大小，在输入 RNet 之前，一律 resize 成 (24, 24)。因为输入的大小是固定的，所以 RNet 中可以使用 FC 层，结果不再是 (H, W, 16) 的 feature map，而是 (16, ) 的向量。同样，代码里的输出的结果只里包含了 2 维的 face classification，4 维的 bounding box regression，并没有输出 10 维的 facial landmark localization。

输入的 (24, 24, 3) 的图片经过 rnet，得到 2 维的 face classification，4 维的 bounding box regression。去掉判定为人脸的概率小于 threshold (0.7) 的，然后将剩下的 bb 做 NMS，最后将得到的 bb 坐标用 regression 中的 offset 精校一下，并填充为正方形。得到 RNet 输出的 bb。

RNet 这一步不再产生新的 bb，而是对 PNet 产生的 bb 坐标的作进一步的精调。

#### ONet
ONet 的输入是 RNet 产生的所有 bb，并且 resize 成 (48, 48)。输出 2 维的 face classification，4 维的 bounding box regression，以及 10 维的 facial landmark localization。

bb 及 regression 的处理方法同上。这里多出了 10 维的 facial landmark localization。从代码里看
```python
points[0:5, :] = np.tile(total_boxes[:, 0], (5, 1)) + np.tile(ww, (5, 1)) * points[0:5, :] - 1
points[5:10, :] = np.tile(total_boxes[:, 1], (5, 1)) + np.tile(hh, (5, 1)) * points[5:10, :] - 1
```

这 10 维是相对于 bb 长宽的偏置，其中前 5 维是 x 坐标偏置，后 5 维是 y 坐标偏置。

至此，我们就得到了一张图片中人脸框的坐标和五个点的坐标。下面的工作就是把 bb 从原图中抠出来，这里就不详述了。

### 4. MTCNN TensorFLow Serving
需要将 mtcnn 中建立的 pnet/rnet/onet 保存下来，并且转换成 tensorflow serving 可用的格式，然后起一个 tensorflow_model_server 来运行 model。

#### 使用 tf.train.Saver() 保存模型
代码里需要保存的文件有两个：metagraph (model.meta) 文件和 checkpoint (model.ckpt) 文件。实际生成的文件有 4 个：(model.meta) 文件保存了 metagraph 信息，即计算图的结构；(model.ckpt.data) 文件保存了 graph 中的所有变量的数据；(model.ckpt.index) 保存了如何将 meta 和 data 匹配起来的信息；(checkpoint) 文件保存了文件的绝对路径，告诉 TF 最新的 ckpt 是哪个，保存在哪里，在使用 tf.train.latest_checkpoint 加载的时候要用到这些信息，但是如果改变或者删除了文件中保存的路径，那么加载的时候会出错，找不到文件。

#### 使用 tf.saved_model.builder.SavedModelBuilder() 保存模型
使用 tf.train.Saver() 保存的模型在 TF serving 上不能用，因此需要将上述模型用 SavedModelBuilder 来 export 成 TF serving model。这里仅通过 mtcnn 这个例子来看看如何 export 一个 model，SavedModelBuilder 更复杂的用法，以后见到再见招拆招吧。

首先要将上述保存的模型加载进来，然后通过名字来找到输入输出 tensor，并写入模型的 signature 中。最后用 SavedModelBuilder 将 graph 和 data 匹配起来保存。保存生成的文件有三个，(saved_model.pb) 是模型的 protobuf 文件；模型的数据保存在 variables 文件夹下的 (variables.data) 和 (variables.index) 两个文件中。

#### TF serving
```
sudo tensorflow_model_server --port=9000 --enable_batching=true --model_config_file=/home/lzhang/model_zoo/TensorFlow/mtcnn/model.config
```
启动 TF serving 服务的时候，要为 model 建一个 config 文件，里面写明了 model 的路径和名称。在 client 调用服务的时候，要用到这里制定的 model 的名称和上一步定义的 signature。
