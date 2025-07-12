# Goal of the project
This project was my Thesis work for my BSc Diploma.
The aim was to create an AI agent system that can execute different tasks given in natural language in a video game environment or in this case Minecraft.

# Parts of the system
There are three main parts of the system. Autoencoder, position transformer and RL transformer.
In my thesiswork the first two parts were properly developed but the RL transformer was not completed because of lack of time.

In this project I aim to recreate the models, improve my code and finish the last model.

## Autoencoder
This model is a CNN based autoencoder model. During training images from minecraft gameplays are fed to the model. In the encoder part of the model the images go through the convolution layers and the the result is vectorized. It is the latent space and the goal of this model is to encode the images in a meaningful way.

The second part of the model is the decoder. It aims to restore the original image from the vectorized version with minimal loss. The architecture of the decoder is similar to the encoder but symmetric to it.