# Advanced Machine Learning

This repository contains information about the course on Advanced Data
Analysis and Machine Learning, spanning from weekly plans to lecture
material and various reading assignments.  The emphasis is on deep
learning algorithms, starting with the mathematics of neural networks
(NNs), moving on to convolutional NNs (CNNs) and recurrent NNs (RNNs),
autoencoders, transformers, graph neural networks and other dimensionality reduction methods to finally
discuss generative methods. These will include Boltzmann machines,
variational autoencoders, generalized adversarial networks, diffusion methods and other.
Reinforcement learning is another topic which can be covered if there is enough interest.

![alt text](https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/images/image001.jpg?raw=true)


## Practicalities

- Lectures Thursdays 1015am-12pm, room FØ434, Department of Physics
- Lab and exercise sessions Thursdays 1215pm-2pm, , room FØ434, Department of Physics
- We plan to work on two projects which will define the content of the course, the format can be agreed upon by the participants
- No exam, only two projects. Each projects counts 50% of the final grade. Alternatively one long project which counts 100% of the final grade.
- All info at the GitHub address https://github.com/CompPhysics/AdvancedMachineLearning
- Permanent Zoom link for the whole semester is https://uio.zoom.us/my/mortenhj
  
## Deep learning methods covered (tentative plan)

### Deep learning, classics
- Feed forward neural networks and its mathematics (NNs)
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs)
- Autoencoders and principal component analysis
- Transformers (tentative, if interest)

### Deep learning, generative methods
- Basics of generative models
- Boltzmann machines and energy based methods
- Diffusion models
- Variational autoencoders (VAEe)
- Autoregressive methods
- Generative Adversarial Networks (GANs)
- Normalizing flows (tentative, if interest)


### Reinforcement Learning
- Basics of reinforcement learning

### Physical Sciences (often just called Physics informed) informed machine learning
- Basic set up of PINNs with discussion of projects


All teaching material is available from this GitHub link.


The course can also be used as a self-study course and besides the
lectures, many of you may wish to independently work on your own
projects related to for example your thesis or research. In general,
in addition to the lectures, we have often followed five main paths:

- Projects (two in total)  and exercises that follow the lectures

- The coding path. This leads often to a single project only where one focuses on coding for example CNNs or RNNs or parts of LLMs from scratch.

- The Physics Informed neural network path (PINNs). Here we define some basic PDEs which are solved by using PINNs. We start normally with studies of selected differential equations using NNs, and/or RNNs, and/or GNNs or Autoencoders before moving over to PINNs. 

- The own data path. Some of you may have data you wish to analyze with different deep learning methods

- The Bayesian ML path is not covered by the present lecture material. It is  normally based on independent work.



## January 19-23: Presentation of couse, review of neural networks and deep Learning and discussion of possible projects

- Presentation of course and overview
- Discussion of possible projects
- Deep learning, neural networks, basic equations
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week1/ipynb/week1.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka chapter 11
- Video of lecture at https://youtu.be/

##  January 26-30
- Mathematics  of deep learning, basics of neural networks and writing a neural network code
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week2/ipynb/week2.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka et al chapter 11. For Pytorch see Raschka et al chapter 12.
- Link to video of lecture  at https://youtu.be/



## February 2-6
- From neural networks to convolutional neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week3/ipynb/week3.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka et al chapter 11. For Pytorch see Raschka et al chapter 12.

## February 9-13
- Mathematics of convolutional neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week4/ipynb/week4.ipynb
- Recommended reading Goodfellow et al chapter 9. Raschka et al chapter 13
- Video of lecture at https://youtu.be/



## February 16-20
- Mathematics of  CNNs and discussion of codes
- Recurrent neural networks (RNNs)
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week5/ipynb/week5.ipynb
- Recommended reading Goodfellow et al chapter 9. Raschka et al chapter 13
- Video of lecture at https://youtu.be/

## February 23-27
- Mathematics of recurrent neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week6/ipynb/week6.ipynb
- Recommended reading Goodfellow et al chapters 9 and 10 and Raschka et al chapters 14 and 15
- Video of lecture at https://youtu.be/


## March 2-6
- Recurrent neural networks, mathematics  and codes
- Applications to differential equations
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week7/ipynb/week7.ipynb
- Recommended reading Goodfellow et al chapters 10 and Raschka et al chapter 15 and 18
- Video of lecture at https://youtu.be/

## March 9-13
- Long short term memory and RNNs
- Autoencoders and PCA
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week8/ipynb/week8.ipynb
- Recommended reading Goodfellow et al chapter 14 for Autoenconders and Rashcka et al chapter 18
- Video	of lecture at https://youtu.be/


## March 16-20: Autoencoders
- Autoencoders and links with Principal Component Analysis. Discussion of AE implementations
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week9/ipynb/week9.ipynb- Reading recommendation: Goodfellow et al chapter 14
- Video of Lecture at https://youtu.be/


## March 23-27: Generative models
- Monte Carlo methods and structured probabilistic models for deep learning
- Partition function and Boltzmann machines
- Boltzmann machines
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week10/ipynb/week10.ipynb  
- Reading recommendation: Goodfellow et al chapters 16-18
- Video of lecture at https://youtu.be/

## March 30- April 3: Public holiday, no lectures

## April 6-10: Deep generative models, Boltzmann machines
- Restricted Boltzmann machines
- Reminder on Markov Chain Monte Carlo and Gibbs sampling
- Discussions of various Boltzmann machines
- Reading recommendation: Goodfellow et al chapters 16, 17 and 18
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week11/ipynb/week11.ipynb  


## April 13-17: Deep generative models
- Reminder from previous week on Energy-based models and Langevin sampling
- Variational Autoencoders
- Reading recommendation: Goodfellow et al chapters 18.1-18.2,  20.1-20-7; To create Boltzmann machine using Keras, see Babcock and Bali chapter 4
- See also Foster, chapter 7 on energy-based models
- Video of lecture at https://youtu.be/

## April 20-24: Deep generative models

- Variational autoencoders
- Reading recommendation: Goodfellow et al chapter 20.10-20.14
- See also Foster, chapter 7 on energy-based models
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week13/ipynb/week13.ipynb  
- Video of lecture at https://youtu.be/


## April 27 - May 1:   Deep generative models
- Diffusion models
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week14/ipynb/week14.ipynb  
- Video	of lecture at https://youtu.be/


## May 4-8: Deep generative models
- Diffusion models
- GANs
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week15/ipynb/week15.ipynb  
- Video of lecture at https://youtu.be/


## May 11-15:  Discussion of projects and summary of course
- Summary slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week16/ipynb/week16.ipynb  


## Recommended textbooks:

o Goodfellow, Bengio and Courville, Deep Learning at https://www.deeplearningbook.org/

o Sebastian Raschka, Yuxi Lie, and Vahid Mirjalili,  Machine Learning with PyTorch and Scikit-Learn at https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312, see also https://sebastianraschka.com/blog/2022/ml-pytorch-book.html

o David Foster, Generative Deep Learning, https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/

o Babcock and Gavras, Generative AI with Python and TensorFlow, https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2

o Sutton and Barto, An Introduction to Reinforcement Learning, https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
