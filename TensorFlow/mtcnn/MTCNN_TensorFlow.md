MTCNN 的 TensorFlow 实现及 serving
---
Magic Vision 要加人脸识别功能，所以要在 TensorFlow Serving 上起一个人脸识别服务，自然想到的是 Google 的 [Facenet](https://github.com/davidsandberg/facenet)。 由于 Google 官方提供下载的 Facenet 模型中有个 [bug](https://github.com/davidsandberg/facenet/issues/789) 导致其 serving 不起来，所以要在他们的源代码上进行修改重新训练。于是乎我想干脆把他们的代码自己重新写一遍算了。至于如何从头开始训练 Facenet，参见这篇 [文章](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)。

Facenet 的实现过程包括两步，首先是用 MTCNN 将图片中的人脸框出来，第二步是识别框出来的人脸是谁。这里先完成第一步，即 MTCNN 的 TensorFlow 实现，并将得到的 model 在 TensorFlow Serving 上跑起来。

MTCNN [原始论文](https://arxiv.org/abs/1604.02878) 中的 [代码](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 是用 MATLAB 实现的。Facenet 只是将 MATLAB 代码翻译成了 TensorFlow , 并使用已经训练好的模型参数。这里不得不吐槽一下原始论文中的 MATLAB 代码，到处都是多余的 T 啊，有事没事的就来个转置，这肯定是平时写论文推公式养成的习惯，见到个矩阵后面就加个 T，本来好好的 (x, y) 坐标，非得要加个 T 变成 (y, x)，完事再 T 回来。Google 也耿直，翻译代码的时候也是见到 T 就 np.transpose。我试着将多余的转置去掉，发现不行，结果不对，可能现成的参数就是这么训练出来的，如果去掉转置的话，参数可能对不上。长话短说，还是先看代码吧。

### Dataset : CASIA-maxpy-clean
Facenet 用的是 CASIA-webface dataset 进行训练。这个 dataset 在原始地址已经下载不到了，而且这个 dataset 据说有很多无效的图片，所以我用的是清理过的数据库。该数据库在百度网盘有下载：[下载地址](http://pan.baidu.com/s/1kUdRRJT)，提取密码为 3zbb。

这个数据库有 10575 个类别，每个类别都有各自的文件夹，里面有同一个人的几张或者几十张不等的脸部图片。MTCNN 的工作就是从这些照片中把人物的脸框出来，然后交给下面的 Facenet 去处理。这里建立一个 ImageClass，存储各个类别的编号名称和该类别下所有图片的绝对路径。

### 建立模型
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





























end
