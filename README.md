# LSGAN

LSGAN is a GAN model that improves training stability through changes in loss function in vanilla GAN

The advantage of this model is that the generated image quality is good and easy to implement because it is equivalent to giving an l2 loss for the distribution of training data and the distribution of generated data

But like most GAN models, they have an inherent training instability problem

For example, typically mode collapse and oscillation

These problems arise due to the architecture of the basic GAN, because the generator model that generates the image and the discriminator model that determines the authenticity of the image do not balance the training

This repository mitigates the above issue by placing a simple loss constraint on the discriminator

If the loss of the discriminator converges below a certain level, the corresponding termination ignores the discriminator's learning

This approach, which may seem simple, has improved the training stability of LSGAN

Below is the training progress of the actual LSGAN trained mnists and fashion mnists

<img src="/md/mnist.gif" width="420"><br>

<img src="/md/fashion_mnist.gif" width="420"><br>
