Facenet 的 TensorFlow 实现
---
上一节使用 mtcnn 可以将图片中的人的面部图像切割出来，这一节就要捕捉这些不同人的面部图像的特征，实现人脸识别。这一节参考的是 [Google Facenet](https://github.com/davidsandberg/facenet)，其理论基础是文章
[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://github.com/FortiLeiZhang/model_zoo/blob/master/TensorFlow/facenet/doc/FaceNet%20A%20Unified%20Embedding%20for%20Face%20Recognition%20and%20Clustering.pdf)。这里我们的目标是重写 facenet，将生成的模型导出并 TF Serving 起来。

在参考的 facenet 实现中，与原文不同的是，作者使用了 softmax loss 来代替 triplet loss，这里就先从 train_softmax 开始，完成后再研究 train_triplet。使用 softmax loss 训练 facenet 的步骤可以参考 [链接](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)。下文将按照 train_softmax.py 代码的顺序来写。

### 1. LFW 测试
LFW (Labeled Face in the Wild) 测试集选择了 6000 对人脸组成了人脸辨识图片对，其中 3000 对属于同一个人的 2 张人脸照片，3000 对属于不同的两个人的，每人 1 张人脸照片。测试过程 LFW 给出一对照片，询问测试中的系统两张照片是不是同一个人，系统给出 "是" 或者 "否" 的答案。通过 6000 对人脸测试结果的系统答案与真实答案的比值可以得到人脸识别准确率。所以在给出的 pairs.txt 中，
```
Bill_Frist	2	9

Larry_Ralston	1	William_Donaldson	6
```
第一行表示同一个人的第 2 和第 9 张照片，第二行表示两人的照片。所以代码中
```python
if args.lfw_dir:
    print('LFW directory: %s' % args.lfw_dir)
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
```
lfw_paths 和 actual_issame 是两个 list，lfw_paths 中存储的是照片对的路径，其对应位置的 actual_issame 存储的是两张照片是否是同一个人。

### 2. index queue
这里首先建立了 image_list 和 label_list，其中 image_list 里包含了 dataset 中所有图片的路径，label_list 中包含了图片对应的编码，同一人的图片编码相同。然后
```python
index_q = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)
index_deq_op = index_q.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')
```
生成了一个 index queue，大小是图片的数量，但是这里 dequeue 的时候，数量是一个 epoch 中用到的图片的数量，两者可能不等。

### 3. input queue
input queue 中一组数据包括了 image path，label，由于在训练的时候需要对图片做一下 augmentation，所以还加入了 control 数据。 如果是做 inference 的话，label 和 control 是不是可以省掉？这个都弄完了回头再来看。

### 4. create_input_pipeline
这里将 input_queue 中的 (image, label, control) 元祖 dequeue 出来，根据 control 里的内容对 image 进行各种预处理，然后将处理后的 (image, label) 打包成真正输入 model 的 batch。

### 5. 建立 NN
这里使用的是 inception_resnet_v1，具体 NN 的细节这里就不多说了，一层层垒就好了，这里看一下 NN 的输出。
```python
with tf.variable_scope('Logits'):
    end_points['PrePool'] = net
    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
    net = slim.flatten(net)
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
    end_points['PreLogitsFlatten'] = net

net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
```
NN 的 dropout 发生在最后一层 FC 之前，FC 层的输出是形如 (N, bottleneck_layer_size)，也就是 (N, embedding_size)。这里 embedding_size 取的是 512，也就是 NN 输出的是每一张图片 512 维的特征值。对上述输出的特征值进行 l2_normalize，就得到了每一张图片的 embeddings。
```python
embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
```
同时，再对 NN 输出的特征值做一次 FC，将其投射成 (N, C)，其中 C 是整个 train_set 中所包含的类别数，那么就得到了各张图片在各个类别的 logit。
```python
logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, reuse=False, scope='Logits',
                             weights_initializer=slim.initializers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(args.weight_decay))
```

### 6. loss function
对 logits 作 softmax 可以得到 cross entropy loss；对 prelogits 取 norm，可以得到 reg loss；facenet 这里还多了另外一项对 prelogits 的 center loss。把这三项加起来就是 total loss。






















end
