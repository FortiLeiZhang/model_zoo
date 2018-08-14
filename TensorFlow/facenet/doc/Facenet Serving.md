Facenet Serving
---
训练好的模型最终还是要放到 TF Serving 上来跑。原作者给出了几个训练好的 [pretrain model](https://github.com/davidsandberg/facenet/wiki#pre-trained-models)，同时，在 [download_and_extract.py](https://github.com/davidsandberg/facenet/blob/master/src/download_and_extract.py) 中，还能发现几个老版本的 model 可以下载。然而，原作者在代码中用到了 tf.py_func 这一函数，但是 TF Serving 并不支持这个函数。所以，要想在 TF Serving 上使用原作者的 model，还需要做一些额外的工作把包含 tf.py_func 的 tensor 删除。开始，我在源代码中将包含 tf.py_func 的函数删掉，并且按照作者给出的教程，重新训练了这个模型，但是训练过程时间太长，不值得。更为简捷的方法是，建立一个和作者一样的 Graph，然后从作者给出的 pretrain model 中拷贝所有的 trainable_variables 到这个 Graph 并保存。具体代码参见 [copy_weights_from_existing_model.py](https://github.com/FortiLeiZhang/model_zoo/blob/master/TensorFlow/facenet/src/copy_weights_from_existing_model.py)。

接下来的工作就是 export model，写 client 脚本，将服务在 TF Serving 上跑起来，有了 MTCNN Serving 的经验，这里就不多说了。

还有一点要注意的是，因为 TF Serving 的输入要求是 base64 string，所以 client 传给 server 的参数是 tf.string，然后在 server 端调用 tf.image.decode_image 将 string 解码成 np array。但是，Python 2 和 Python 3 在将图片读取为 string 时的结果是不一样的，其结果就是 Python 2 读取的图片 string 可以经过 b64encode 后被 tf.image.decode_image 识别解码；但是 Python 3 就不可以。这个现象很奇怪，虽然我在这个 [issue](https://github.com/tensorflow/tensorflow/issues/21547) 中找到了一个解决的办法，但是后面遇到同样的问题是，这个方法好像并不是每次都管用。






























end
