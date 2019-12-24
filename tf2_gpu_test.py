import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
#print('gpu',tf.config.list_physical_devices('GPU'))
print('GPU', tf.test.is_gpu_available())

a = tf.constant(np.random.random((10000,5000)))
b = tf.constant(np.random.random((5000,3000)))
print(tf.matmul(a,b))


