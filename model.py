"""
Authors : inzapp

Github url : https://github.com/inzapp/lsgan

Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, generate_shape, latent_dim):
        self.generate_shape = generate_shape
        self.latent_dim = latent_dim
        self.gan = None
        self.g_model = None
        self.d_model = None
        self.latent_rows = generate_shape[0] // 8
        self.latent_cols = generate_shape[1] // 8
        self.latent_channels = 8192 // (self.latent_rows * self.latent_cols)
        if self.latent_channels > 128:
            self.latent_channels = 128

    def build(self):
        assert self.generate_shape[0] % 32 == 0 and self.generate_shape[1] % 32 == 0
        assert self.generate_shape[0] <= 256 and self.generate_shape[1] <= 256
        g_input, g_output = self.build_g(bn=True)
        d_input, d_output = self.build_d(bn=False)
        self.g_model = tf.keras.models.Model(g_input, g_output)
        self.d_model = tf.keras.models.Model(d_input, d_output)
        gan_output = self.d_model(g_output)
        self.gan = tf.keras.models.Model(g_input, gan_output)
        return self.g_model, self.d_model, self.gan

    def build_g(self, bn):
        g_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = g_input
        x = self.dense(x, 1024, activation='leaky', bn=bn)
        x = self.dense(x, 2048, activation='leaky', bn=bn)
        x = self.dense(x, self.latent_rows * self.latent_cols * 128, activation='leaky', bn=bn)
        x = self.reshape(x, (self.latent_rows, self.latent_cols, self.latent_channels))
        x = self.conv2d_transpose(x, 128, 1, 1, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 128, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 128, 3, 1, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x,  64, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x,  64, 3, 1, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x,  32, 3, 2, activation='leaky', bn=bn)
        g_output = self.conv2d_transpose(x, self.generate_shape[-1], 1, 1, activation='tanh', bn=False)
        return g_input, g_output

    def build_d(self, bn):
        d_input = tf.keras.layers.Input(shape=self.generate_shape)
        x = d_input
        x = self.conv2d(x,  32, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x,  64, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 128, 3, 1, activation='leaky', bn=bn)
        x = self.conv2d(x, 128, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 256, 3, 1, activation='leaky', bn=bn)
        x = self.conv2d(x, 256, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 512, 3, 1, activation='leaky', bn=bn)
        x = self.conv2d(x, 512, 3, 2, activation='leaky', bn=bn)
        x = self.flatten(x)
        d_output = self.dense(x, 1, activation='linear', bn=False)
        return d_input, d_output

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='leaky'):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides, bn=False, activation='leaky'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='leaky'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=False if bn else True,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization(momentum=0.8)(x)

    def kernel_initializer(self):
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def activation(self, x, activation):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.g_model.summary()
        print()
        self.gan.summary()

