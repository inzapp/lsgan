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
import numpy as np
import tensorflow as tf

import cv2
from glob import glob
from time import time
from model import Model
from lr_scheduler import LRScheduler
from generator import DataGenerator


class LSGAN:
    def __init__(self,
                 train_image_path,
                 generate_shape=(32, 32, 1),
                 lr=1e-3,
                 batch_size=32,
                 latent_dim=32,
                 save_interval=2000,
                 iterations=100000,
                 view_grid_size=4,
                 d_loss_ignore_threshold=0.1,
                 checkpoint_path='checkpoint',
                 training_view=False):
        assert generate_shape[2] in [1, 3]
        self.generate_shape = generate_shape
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval
        self.iterations = iterations
        self.view_grid_size = view_grid_size
        self.training_view = training_view
        self.d_loss_ignore_threshold = d_loss_ignore_threshold
        self.checkpoint_path = checkpoint_path
        self.live_view_previous_time = time()

        self.model = Model(generate_shape=generate_shape, latent_dim=self.latent_dim)
        self.g_model, self.d_model, self.gan = self.model.build()

        self.train_image_paths = self.init_image_paths(train_image_path)
        self.train_data_generator = DataGenerator(
            generator=self.g_model,
            image_paths=self.train_image_paths,
            generate_shape=generate_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def compute_gradient(self, model, optimizer, x, y_true, ignore_threshold):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
            if loss < ignore_threshold:
                loss = 0.0
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def build_loss_str(self, iteration_count, d_loss, g_loss):
        loss_str = f'[iteration_count : {iteration_count:6d}]'
        loss_str += f' d_loss: {d_loss:>8.4f}'
        loss_str += f', g_loss: {g_loss:>8.4f}'
        return loss_str

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        os.makedirs(self.checkpoint_path, exist_ok=True)
        g_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        d_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        compute_gradient_d = tf.function(self.compute_gradient)
        compute_gradient_g = tf.function(self.compute_gradient)
        g_lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='step')
        d_lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='step')
        while True:
            for dx, dy, gx, gy in self.train_data_generator:
                g_lr_scheduler.update(g_optimizer, iteration_count)
                d_lr_scheduler.update(d_optimizer, iteration_count)
                self.d_model.trainable = True
                d_loss = compute_gradient_d(self.d_model, d_optimizer, dx, dy, self.d_loss_ignore_threshold)
                self.d_model.trainable = False
                g_loss = compute_gradient_g(self.gan, g_optimizer, gx, gy, 0.0)
                iteration_count += 1
                print(self.build_loss_str(iteration_count, d_loss, g_loss))
                if self.training_view:
                    self.training_view_function()
                if iteration_count % self.save_interval == 0:
                    model_path_without_extention = f'{self.checkpoint_path}/generator_{iteration_count}_iter' 
                    self.g_model.save(f'{model_path_without_extention}.h5', include_optimizer=False)
                    generated_images = self.generate_image_grid(grid_size=21 if self.latent_dim == 2 else 10)
                    cv2.imwrite(f'{model_path_without_extention}.jpg', generated_images)
                    print(f'[iteration count : {iteration_count:6d}] model with generated images saved with {model_path_without_extention} h5 and jpg\n')
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    while True:
                        generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
                        cv2.imshow('generated_images', generated_images)
                        key = cv2.waitKey(0)
                        if key == 27:
                            exit(0)

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    def generate_random_image(self, size=1):
        z = np.asarray([DataGenerator.get_z_vector(size=self.latent_dim) for _ in range(size)])
        y = np.asarray(self.graph_forward(self.g_model, z))
        generated_images = DataGenerator.denormalize(y).reshape((size,) + self.generate_shape)
        return generated_images[0] if size == 1 else generated_images

    def generate_latent_space_2d(self, split_size=10):
        assert split_size > 1
        assert self.latent_dim == 2
        space = np.linspace(-1.0, 1.0, split_size)
        z = []
        for i in range(split_size):
            for j in range(split_size):
                z.append([space[i], space[j]])
        z = np.asarray(z).reshape((split_size * split_size, 2)).astype('float32')
        y = np.asarray(self.graph_forward(self.g_model, z))
        generated_images = DataGenerator.denormalize(y).reshape((split_size * split_size,) + self.generate_shape)
        return generated_images

    def predict(self, img):
        img = DataGenerator.resize(img, (self.generate_shape[1], self.generate_shape[0]))
        x = np.asarray(img).reshape((1,) + self.generate_shape)
        x = DataGenerator.normalize(x)
        y = np.asarray(self.graph_forward(self.gan, x)[0]).reshape(self.generate_shape)
        decoded_img = DataGenerator.denormalize(y)
        return img, decoded_img

    def show_interpolation(self, frame=100):
        space = np.linspace(-1.0, 1.0, frame)
        for val in space:
            z = np.zeros(shape=(1, self.latent_dim), dtype=np.float32) + val
            y = np.asarray(self.graph_forward(self.g_model, z))[0]
            y = DataGenerator.denormalize(y)
            generated_image = np.clip(np.asarray(y).reshape(self.generate_shape), 0.0, 255.0).astype('uint8')
            cv2.imshow('interpolation', generated_image)
            key = cv2.waitKey(10)
            if key == 27:
                break

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=(192, 192, 192)) 

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 3.0:
            generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
            cv2.imshow('generated_images', generated_images)
            cv2.waitKey(1)
            self.live_view_previous_time = cur_time

    def generate_image_grid(self, grid_size):
        if self.latent_dim == 2:
            generated_images = self.generate_latent_space_2d(split_size=grid_size)
        else:
            generated_images = self.generate_random_image(size=grid_size * grid_size)
        generated_image_grid = None
        for i in range(grid_size):
            grid_row = None
            for j in range(grid_size):
                generated_image = self.make_border(generated_images[i*grid_size+j])
                if grid_row is None:
                    grid_row = generated_image
                else:
                    grid_row = np.append(grid_row, generated_image, axis=1)
            if generated_image_grid is None:
                generated_image_grid = grid_row
            else:
                generated_image_grid = np.append(generated_image_grid, grid_row, axis=0)
        return generated_image_grid

    def show_generated_images(self):
        while True:
            generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
            cv2.imshow('generated_images', generated_images)
            key = cv2.waitKey(0)
            if key == 27:
                break

