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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

import cv2
from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self,
                 train_image_path,
                 generate_shape,
                 lr,
                 batch_size,
                 latent_dim,
                 save_interval,
                 iterations,
                 view_grid_size,
                 model_name,
                 pretrained_g_model_path='',
                 pretrained_d_model_path='',
                 training_view=False):
        self.train_image_path = train_image_path
        self.generate_shape = generate_shape
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval
        self.iterations = iterations
        self.view_grid_size = view_grid_size
        self.model_name = model_name
        self.pretrained_g_model_path = pretrained_g_model_path
        self.pretrained_d_model_path = pretrained_d_model_path
        self.training_view = training_view


class LSGAN(CheckpointManager):
    def __init__(self, config):
        super().__init__()
        assert config.generate_shape[0] % 32 == 0
        assert config.generate_shape[1] % 32 == 0
        assert config.generate_shape[2] in [1, 3]
        self.train_image_path = config.train_image_path
        self.generate_shape = config.generate_shape
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.save_interval = config.save_interval
        self.iterations = config.iterations
        self.view_grid_size = config.view_grid_size
        self.model_name = config.model_name
        self.pretrained_g_model_path = config.pretrained_g_model_path
        self.pretrained_d_model_path = config.pretrained_d_model_path
        self.training_view = config.training_view

        self.set_model_name(self.model_name)
        warnings.filterwarnings(action='ignore')

        if self.pretrained_g_model_path == '' and self.pretrained_d_model_path == '':
            self.model = Model(generate_shape=self.generate_shape, latent_dim=self.latent_dim)
            self.g_model, self.d_model, self.gan = self.model.build()
        else:
            pretrained_g_model = None
            pretrained_d_model = None
            if self.pretrained_g_model_path != '':
                if os.path.exists(self.pretrained_g_model_path) and os.path.isfile(self.pretrained_g_model_path):
                    pretrained_g_model = tf.keras.models.load_model(self.pretrained_g_model_path, compile=False)
                    self.g_model = pretrained_g_model
                    self.latent_dim = self.g_model.input_shape[1:]
                    self.generate_shape = self.g_model.output_shape[1:]
                else:
                    print(f'g_model file not found : {self.pretrained_g_model_path}')
                    exit(0)

            if self.pretrained_d_model_path != '':
                if os.path.exists(self.pretrained_d_model_path) and os.path.isfile(self.pretrained_d_model_path):
                    pretrained_d_model = tf.keras.models.load_model(self.pretrained_d_model_path, compile=False)
                    self.d_model = pretrained_d_model
                else:
                    print(f'd_model file not found : {self.pretrained_d_model_path}')
                    exit(0)

            self.model = Model(generate_shape=self.generate_shape, latent_dim=self.latent_dim)
            self.g_model, self.d_model, self.gan = self.model.build(g_model=pretrained_g_model, d_model=pretrained_d_model)

        self.train_image_paths = self.init_image_paths(self.train_image_path)
        self.train_data_generator = DataGenerator(
            generator=self.g_model,
            image_paths=self.train_image_paths,
            generate_shape=self.generate_shape,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def print_loss(self, progress_str, d_loss, g_loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' discriminator_loss: {d_loss:>8.4f}'
        loss_str += f', generator_loss: {g_loss:>8.4f}'
        print(loss_str, end='')

    def train(self):
        if len(self.train_image_paths) == 0:
            print(f'no images found in {self.train_image_path}')
            exit(0)

        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        d_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        g_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        compute_gradient_d = tf.function(self.compute_gradient)
        compute_gradient_g = tf.function(self.compute_gradient)
        g_lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='onecycle')
        d_lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='onecycle')
        self.init_checkpoint_dir()
        eta_calculator = ETACalculator(iterations=self.iterations)
        eta_calculator.start()
        g_losses, d_losses = [], []
        while True:
            for dx, dy, gx, gy in self.train_data_generator:
                g_lr_scheduler.update(g_optimizer, iteration_count)
                d_lr_scheduler.update(d_optimizer, iteration_count)
                self.d_model.trainable = True
                d_loss = compute_gradient_d(self.d_model, d_optimizer, dx, dy)
                d_losses.append(d_loss)
                self.d_model.trainable = False
                g_loss = compute_gradient_g(self.gan, g_optimizer, gx, gy)
                g_losses.append(g_loss)
                iteration_count += 1
                progress_str = eta_calculator.update(iteration_count)
                self.print_loss(progress_str, d_loss, g_loss)
                if self.training_view:
                    self.training_view_function()
                if iteration_count % self.save_interval == 0:
                    model_path_without_extention = f'{self.checkpoint_path}/generator_{iteration_count}_iter'
                    self.g_model.save(f'{model_path_without_extention}.h5', include_optimizer=False)
                    generated_images = self.generate_image_grid(grid_size=21 if self.latent_dim == 2 else 10)
                    cv2.imwrite(f'{model_path_without_extention}.jpg', generated_images)
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    # self.plot_loss(d_losses, g_losses, iteration_count)
                    exit(0)

    def plot_loss(self, d_losses, g_losses, iteration_count):
        from matplotlib import pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        x = range(iteration_count)

        g_losses = np.clip(np.array(g_losses).astype('float32'), -0.1, 1.1)
        d_losses = np.clip(np.array(d_losses).astype('float32'), -0.1, 1.1)

        ax1.plot(x, g_losses, 'g-', label='g_loss')
        ax2.plot(x, d_losses, 'b-', label='b_loss')

        fig.legend(loc="upper right")
        ax1.set_xlabel('Iteration')
        plt.tight_layout(pad=0.5)
        plt.grid()
        plt.show()

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    def generate(self, save_count, save_grid_image, grid_size):
        save_dir_path = 'generated_images'
        os.makedirs(save_dir_path, exist_ok=True)
        elements = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
        if save_count > 0:
            for i in tqdm(range(save_count)):
                random_stamp = ''.join(np.random.choice(elements, 12))
                if save_grid_image:
                    save_path = f'{save_dir_path}/generated_grid_{i}_{random_stamp}.jpg'
                    img = self.generate_image_grid(grid_size=grid_size)
                else:
                    save_path = f'{save_dir_path}/generated_{i}_{random_stamp}.jpg'
                    img = self.generate_random_image(size=1)
                cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            while True:
                if save_grid_image:
                    img = self.generate_image_grid(grid_size=grid_size)
                else:
                    img = self.generate_random_image()
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)

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
        if grid_size == 'auto':
            border_size = 10
            grid_size = min(720 // (self.generate_shape[0] + border_size), 1280 // (self.generate_shape[1] + border_size))
        else:
            if type(grid_size) is str:
                grid_size = int(grid_size)
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

