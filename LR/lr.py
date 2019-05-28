#! /usr/bin/env python3
# -*-:utf-8-*-

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

X_raw = np.array([2013, 2014, 2015, 2016, 2017])
y_raw = np.array([12000, 14000, 15000, 16500, 17500])

# 归一化/标准化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

def numpy_default():
    #梯度下降
    a, b = 0, 0
    num_epoch = 10000
    learning_rate = 1e-3

    for e in range(num_epoch):
        # 手动计算损失函数 关于自变量的梯度
        y_pred = a * X + b
        grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

        a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

    print(a, b)

def use_tensorflow():
    a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    variables = [a, b]

    num_epoch = 10000
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    for e in range(num_epoch):
        # 使用 tf.GradientTape() 记录损失函数的梯度信息
        with tf.GradientTape() as tape:
            # 模型
            y_pred = a * X + b
            # 定义损失函数
            loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))

        # tensorflow 自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)

        # Tensorflow 自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    print(a,b)


if __name__ == '__main__':
    use_tensorflow()
