MTCNN TensorFLow Serving
---
上一节用 Keras 实现了 mtcnn，下一步就是在 TF Serving 上将这个服务起来。本来以为很简单，就是把 model 导出来，在 TF model server 上跑起来就完事。TF 官方提供了一个能用的 [参考版本](https://github.com/davidsandberg/facenet/issues/758)。但是，真正用到实际中，问题又来了。首先，在向 serving model 传递输入的时候，使用到了 tf.make_tensor_proto
```python
tp = tf.make_tensor_proto(image, dtype=tf.float32, shape=image.shape)
```
而在实际的应用程序一侧，使用 tf 里面的函数需要 import tensorflow，这会耗费大量的内存。另外，serving model 的输入是 numpy array，这在进行大量调用的时候，效率很低，根据以往在做 object detection 时的经验，传递参数为 bytes 最好。所以要解决的问题为：

1. 改造模型，使其输入为 bytes 而非 numpy array。
2. 在 client 中避免使用 tf.make_tensor_proto。

### 改造 MTCNN 模型
原始的模型输入是形如 (None,None,None,3) 的 tf.float32 数组
```python
data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
```
这里我们的模型输入是形如 (None, 1) 的 tf.string 数组
```python
batch_image_str_placeholder = tf.placeholder(dtype=tf.string, shape=[None, ], name='encoded_image_string_tensor')
```
为了与原模型兼容，下一步就是将上述的 tf.string 数组转换为等价的 tf.float32 数组。这里用到的函数是 tf.image.decode_image。但是，在 tf.image.decode_image 的 [手册说明](https://www.tensorflow.org/api_docs/python/tf/image/decode_image) 中，明确指出了该函数的输入是一个 0-D string，所以需要用到 tf.map_fn 对输入数组的每一个 string 进行处理。
```python
def decode_encoded_image_string_tensor(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor, channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor

pnet_image_tensor = tf.map_fn(decode_encoded_image_string_tensor, elems=batch_image_str_placeholder, dtype=tf.uint8, back_prop=False)
```
同时，由于输入的是图片读取出来的 tf.uint8 的 bytes，所以，原来在 model 外进行的减均值除方差操作要放到 model 里面来进行
```python
pnet_image_tensor = tf.cast(pnet_image_tensor, tf.float32)
pnet_image_tensor = tf.subtract(pnet_image_tensor, 127.5)
pnet_image_tensor = tf.multiply(pnet_image_tensor, 0.0078125)
pnet_image_tensor.set_shape((None, None, None, 3))
```
最后，model 接受 feed_dict 的 tensor 不再是 'pnet/input:0'，而是 'pnet/encoded_image_string_tensor:0'
```python
pnet_fun = lambda img : sess.run(('pnet/p_net/conv4-2/BiasAdd:0', 'pnet/p_net/prob1/truediv:0'), feed_dict={'pnet/encoded_image_string_tensor:0':img})
```
### 导出 MTCNN 模型
与原来模型导出的代码相同，只是要把 feed tensor 的名字改成 'pnet/encoded_image_string_tensor:0'
```python
x_pnet = graph.get_tensor_by_name('pnet/encoded_image_string_tensor:0')
```

### MTCNN client
基本就是仿照以前的 detect_face 来写，只需要改写几个地方。

首先，图片的减均值除方差操作放到 model 里面进行。

其次，原始的 detect_face 输入 model 的参数是 tf.float32 数组，这里需要将其变成 tf.string
```python
def generate_input_string(image):
    image_data_arr = []
    for i in range(image.shape[0]):
        byte_io = BytesIO()
        img = Image.fromarray(image[i, :, :, :].astype(np.uint8).squeeze())

        img.save(byte_io, 'JPEG')
        byte_io.seek(0)
        image_data = byte_io.read()
        image_data_arr.append([image_data])
    return image_data_arr
```
试过几种直接将图片的 tf.float32 数组转换成 tf.string 数组的方法，包括 base64，tobytes() 方法，结果都不对。然后不得不将图片的 tf.float32 数组先保存为图片，再按照 bytes 读进来，这里图片的保存读入都放到内存里进行，应该快点。

最后一步，就是不用 tf.make_tensor_proto 而将 string 数组转换成 feed 进 model 的 tensor。这里参考了这篇[博文](https://towardsdatascience.com/tensorflow-serving-client-make-it-slimmer-and-faster-b3e5f71208fb)。具体方法就是用 protobuf 自己来实现 tf.make_tensor_proto 的功能。
```python
image_data_arr = generate_input_string(image)
image_data_arr = np.asarray(image_data_arr).squeeze(axis=1)

dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
tensor_proto = tensor_pb2.TensorProto(
    dtype=types_pb2.DT_STRING,
    tensor_shape=tensor_shape_proto,
    string_val=[image_data for image_data in image_data_arr])
request.inputs['images'].CopyFrom(tensor_proto)
```

到此，改造 MTCNN 模型的工作就完成了。
