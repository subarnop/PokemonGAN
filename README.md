# PokemonGAN

We are making a DCGAN to generate newer variants of [Pokemon](https://drive.google.com/open?id=1qWuOlnPc4bYRTbKoQDZZWXjo9UcEcPWD)(link to training dataset) from random noise inputs.

The DCGAN used in this project is based on the paper by [Chintala et al.](https://arxiv.org/pdf/1511.06434.pdf)

Generative Adversarial Networks are used for generative modeling, was first proposed by [Goodfellow](https://arxiv.org/abs/1406.2661)
This Convolutional GAN mainly consists of two different networks, the genarator and the discriminator. The Generator tries to generates images from random noise and fools the discriminator in the process.

The Generator model consist of a block of layers that consist of Batchnormalization, upsampling followed by a convolution with relu activation. There are 4 such blocks used only the forth block contains tanh function as the activation of the convolution layer.
The Discriminator model is a simple deep convolution network trying to distinguish between true or fake pokemon images.
The combined model is compiled adam optimizer(learning rate=0.0002), with binary cross entropy loss.
The entire training process for 100000 iterations is shown in this video:
<p align="center">
  [![Training Process](https://img.youtube.com/vi/IBPWMNsy2Z0/0.jpg)](https://www.youtube.com/watch?v=IBPWMNsy2Z0)
</p>
A simpler network for Gneration of images in grayscale is also done.
The training process for the images in grayscale is shown below:
<p align="center">
  <img width="400" height="300" src="https://github.com/Subarno/PokemonGAN/blob/master/output.gif">
</p>
This project is still under progress.
