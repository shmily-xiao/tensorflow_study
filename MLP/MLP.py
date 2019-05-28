#! /usr/bin/env python3
# -*-:utf-8-*-

import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.enable_eager_execution()

class DataLoader(object):

    def __init__(self):

        mnist = read_data_sets("/Users/wangzaijun/workspace/alibaba/PAI/handwritten_digits/")

        # mnist = tf.contrib.learn.datasets.load_dataset("/Users/wangzaijun/workspace/alibaba/PAI/handwritten_digits/")
        # 训练
        self.train_data = mnist.train.images
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        # 验证
        self.eval_data = mnist.test.images
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 多层感知机

        # activation=tf.nn.relu 非线性激活函数ReLU
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)

def train_test_model():
    # 超参数
    num_batches = 1000
    batch_size = 50
    learning_rate = 0.003

    model = MLP()
    data_loader = DataLoader()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    '''
    从DataLoader中随机取一批训练数据；
    将这批数据送入模型，计算出模型的预测值；
    将模型预测值与真实值进行比较，计算损失函数（loss）；
    计算损失函数关于模型变量的导数；•使用优化器更新模型参数以最小化损失函数
    '''
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
            print("batch  %d: loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    num_eval_samples = np.shape(data_loader.eval_labels)[0]
    y_pred = model.predict(data_loader.eval_data).numpy()
    print("test accuracy: %f" % (sum(y_pred == data_loader.eval_labels) / num_eval_samples))
    return model


if __name__ == '__main__':

    model = train_test_model()

    cap = cv2.VideoCapture(0)
    while (1):
        # get a frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = cv2.imread(os.path.join("/Users/wangzaijun/workspace/alibaba/tensorflow/tensorflow_study/test_data", "0.png"))

        gray = cv2.resize(gray, (28, 28))

        image_mat = np.asarray(gray, dtype=np.float32).flatten()

        for i in range(image_mat.size):
            image_mat[i] = 255 - image_mat[i]

        for i in range(image_mat.size):
            if image_mat[i] < 150:
                image_mat[i] = 0


        image = image_mat / image_mat.max(axis=0)
        print(image)

        # show a frame
        cv2.imshow("capture",gray)

        pred = model.predict([image]).numpy()
        print("pred %f" %(pred))
        print ("-----------------")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()






