# Advanced Machine Learning

This repository contains information about the course on Advanced Data
Analysis and Machine Learning, spanning from weekly plans to lecture
material and various reading assignments.  The emphasis is on deep
learning algorithms, starting with the mathematics of neural networks
(NNs), moving on to convolutional NNs (CNNs) and recurrent NNs (RNNs),
autoencoders and other dimensionality reduction methods to finally
discuss generative methods. These will include Boltzmann machines,
variational autoencoders, generalized adversarial networks, diffusion methods and other.

![alt text](https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/images/image001.jpg?raw=true)


### Time: Each Thursday at 1215pm-2pm CET room FØ434, (The sessions will be recorded), first time January 23, 2025
### Lab session: Each Thursday at 215pm-3pm CET, room FØ434

FYS5429 zoom link
https://msu.zoom.us/j/6424997467?pwd=TEhTL0lmTmpGbHlnejZQa1pCdzRKdz09

Meeting ID: 642 499 7467
Passcode: FYS4411


Furthermore, all teaching material is available from this GitHub link.

## January 20-24: Presentation of couse, review of neural networks and deep Learning and discussion of possible projects

- Presentation of course and overview
- Discussion of possible projects
- Deep learning, neural networks, basic equations
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week1/ipynb/week1.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka chapter 11

##  January 27-31
- Mathematics  of deep learning, basics of neural networks and writing a neural network code
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week2/ipynb/week2.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka et al chapter 11. For Pytorch see Raschka et al chapter 12.

## February 3-7
- From neural networks to convolutional neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week3/ipynb/week3.ipynb
- Recommended reading Goodfellow et al chapters 6 and 7 and Raschka et al chapter 11. For Pytorch see Raschka et al chapter 12.

## February 10-14
- Mathematics of convolutional neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week4/ipynb/week4.ipynb
- Recommended reading Goodfellow et al chapter 9. Raschka et al chapter 13
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesFebruary6.pdf


## February 17-21
- Mathematics of  CNNs and discussion of codes
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesFebruary13.pdf
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week5/ipynb/week5.ipynb
- Recommended reading Goodfellow et al chapter 9. Raschka et al chapter 13

## February 24-28
- From CNNs to recurrent neural networks
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week6/ipynb/week6.ipynb
- Recommended reading Goodfellow et al chapters 9 and 10 and Raschka et al chapters 14 and 15
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesFebruary20.pdf

## March 3-7
- Recurrent neural networks, mathematics and codes
- Long-Short-Term memory and applications to differential equations
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week7/ipynb/week7.ipynb
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesFebruary27.pdf
- Recommended reading Goodfellow et al chapters 10 and Raschka et al chapter 15


## March 10-14
- More on structure of RNNs
- Autoencoders and PCA
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week8/ipynb/week8.ipynb
- Recommended reading Goodfellow et al chapter 14 for Autoenconders
- Whiteboard notes https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesMarch5.pdf

## March 17-21: Autoencoders
- Autoencoders and discussions of codes
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week9/ipynb/week9.ipynb  
- Reading recommendation: Goodfellow et al chapter 14
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesMarch12.pdf


## March 24-28: Autoencoders and start discussion of generative models
- Autoencoders and links with Principal Component Analysis. Discussion of AE implementations
- Summary of deep learning methods and links with generative models and discussion of possible paths for project 2
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week10/ipynb/week10.ipynb  
- Reading recommendation: Goodfellow et al chapters, 14 and 16
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesMarch19.pdf

## March 31-April 4: Deep generative models
- Monte Carlo methods and structured probabilistic models for deep learning
- Partition function and Boltzmann machines
- Boltzmann machines
- Reading recommendation: Goodfellow et al chapters 16, 17 and 18
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week11/ipynb/week11.ipynb  
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesApril2.pdf


## April 7-11: Deep generative models
- Restricted Boltzmann machines, reminder from last week
- Reminder on Markov Chain Monte Carlo and Gibbs sampling
- Discussions of various Boltzmann machines
- Energy-based models and Langevin sampling
- Implementation of Boltzmann machines using TensorFlow and Pytorch
- Generative Adversarial Networks (GANs)
- Reading recommendation: Goodfellow et al chapters 18.1-18.2,  20.1-20-7; To create Boltzmann machine using Keras, see Babcock and Bali chapter 4
- See also Foster, chapter 7 on energy-based models
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesApril10.pdf


## April 21-25: Deep generative models

- Generative Adversarial Networks (GANs)
- Variational autoencoders
- Reading recommendation: Goodfellow et al chapter 20.10-20.14
- See also Foster, chapter 7 on energy-based models
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week13/ipynb/week13.ipynb  
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesApril24.pdf

## May 5-9: Deep generative models
- Variational Autoencoders
- Diffusion models
- Reading recommendation: An Introduction to Variational Autoencoders, by Kingma and Welling, see https://arxiv.org/abs/1906.02691 
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week14/ipynb/week14.ipynb  
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesMay8.pdf

## May 12-16: Deep generative models
- Summarizing discussion of VAEs
- Diffusion models
- Summary of course and discussion of projects
- Slides at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/pub/week15/ipynb/week15.ipynb  
- Whiteboard notes at https://github.com/CompPhysics/AdvancedMachineLearning/blob/main/doc/HandwrittenNotes/2025/NotesMay15.pdf



## Recommended textbooks:

o Goodfellow, Bengio and Courville, Deep Learning at https://www.deeplearningbook.org/

o Sebastian Raschka, Yuxi Lie, and Vahid Mirjalili,  Machine Learning with PyTorch and Scikit-Learn at https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312, see also https://sebastianraschka.com/blog/2022/ml-pytorch-book.html

o David Foster, Generative Deep Learning, https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/

o Babcock and Gavras, Generative AI with Python and TensorFlow, https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2

