

import tensorflow as tf
import warnings
import logging

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)



print(tf.config.list_physical_devices('GPU'))


gpu_available = tf.test.is_gpu_available()
print("gpu_available", gpu_available)

is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print("is_cuda_gpu_available", is_cuda_gpu_available)

is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))
print("is_cuda_gpu_min_3", is_cuda_gpu_min_3)



with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)


print(c)

print("TF Version:",tf.__version__)




