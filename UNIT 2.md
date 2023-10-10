# NN2

Created: October 10, 2023 1:03 PM

### UNIT-II Introduction - Machine Learning and Deep Learning: Representation Learning, Width and Depth of Neural Networks, Activation Functions: RELU, LRELU, ERELU, Unsupervised Training of Neural Networks, Restricted Boltzmann Machines, Auto Encoders

### Representation Learning

![image](https://github.com/allelbhagya/nn/assets/80905783/7ccbf3b7-32d1-4abd-b569-be6c9a31c9fe)


Representation learning or feature learning is the subdiscipline of the machine learning space that deals with extracting features or understanding the representation of a dataset.

A machine learning paradigm where the algorithm learns a set of features or representations directly from the raw data, without relying on manually engineered features. The goal is to automatically discover and create a meaningful and informative representation of the input data that can be used for various tasks such as classification, regression, or clustering.

1. **Feature Learning:** Representation learning is often synonymous with feature learning. In traditional machine learning, features are typically handcrafted by domain experts. In representation learning, the algorithm automatically learns relevant features from the data.
2. **Unsupervised Learning:** Representation learning often involves unsupervised learning, where the algorithm tries to learn patterns and structures in the data without explicit labels. Unsupervised methods include autoencoders, restricted Boltzmann machines, and various clustering techniques.
3. **Autoencoders:** Autoencoders are a type of neural network used in representation learning. They consist of an encoder network that maps input data to a lower-dimensional representation and a decoder network that reconstructs the input data from this representation.
4. **Transfer Learning:** Representation learning is often used in transfer learning, where a model trained on one task is adapted to another related task. The learned representation captures general features that can be useful across different tasks.
5. **Dimensionality Reduction:** Representation learning can also be viewed as a form of dimensionality reduction, where the goal is to capture the most important information in a lower-dimensional space.

### Width and Depth of a neural network

![image](https://github.com/allelbhagya/nn/assets/80905783/3acbdeab-c9bb-4e87-bdeb-cf55422402de)


**Width** refers to the numb0er of neurons in each layer of a neural network. A wider network has more neurons, which gives it the ability to learn more complex relationships between features. However, wider networks also have more parameters, which can make them more difficult to train and prone to overfitting.

**Depth** refers to the number of layers in a neural network. A deeper network has more layers, which allows it to learn more complex hierarchical representations of data. However, deeper networks can also be more difficult to train and optimize.

## Activation Functions: RELU, LRELU, ERELU

Activation Functions: RELU, LRELU, ERELU

**ReLU** (Rectified Linear Unit) is a non-linear activation function that has become the most popular choice for deep neural networks. It is defined as follows:

`ReLU(x) = max(0, x)`

ReLU has a number of advantages over other activation functions, such as:

- It is computationally efficient, as it only requires a single comparison.
- It has a non-zero gradient for all positive inputs, which helps to avoid the vanishing gradient problem.
- It is sparse, meaning that only a small fraction of neurons are activated at any given time. This can lead to faster training and better generalization performance.

**LReLU** (Leaky ReLU) is a variant of ReLU that addresses one of its limitations: ReLU can cause dead neurons, which are neurons that are never activated. This can happen because ReLU sets all negative inputs to zero. LReLU addresses this problem by allowing a small positive gradient for negative inputs.

`LReLU(x) = max(α * x, x)`

where α is a small positive hyperparameter.

**ERLU** (Exponential Linear Unit) is another variant of ReLU that addresses the dead neuron problem. ERLU uses an exponential function to smooth out the gradient for negative inputs.

`ERLU(x) = x if x > 0 else α * (exp(x) - 1)`

where α is a small positive hyperparameter.

**Comparison of ReLU, LReLU, and EReLU**

The following table summarizes the key differences between ReLU, LReLU, and EReLU:

| Activation function | Definition | Advantages | Disadvantages |
| --- | --- | --- | --- |
| ReLU | ReLU(x) = max(0, x) | Computationally efficient, non-zero gradient for all positive inputs, sparse | Can cause dead neurons |
| LReLU | LReLU(x) = max(α * x, x) | Addresses the dead neuron problem in ReLU | Requires additional hyperparameter to tune |
| EReLU | ERLU(x) = x if x > 0 else α * (exp(x) - 1) | Addresses the dead neuron problem in ReLU, smooth gradient for negative inputs | Requires additional hyperparameter to tune |

## Other Activation Functions

1. **Linear Activation Function:**
    - The linear activation function is the simplest one.
    - $f(x) = x$
    - It produces an output that is proportional to the input. However, it is rarely used in hidden layers of neural networks because it doesn't introduce non-linearity, and the network would essentially be a linear model.
2. **Sigmoid Activation Function:**
    - The sigmoid function is a type of logistic function.
    - $f(x) = \frac{1}{1 + e^{-x}}$
    - The sigmoid function maps any input to a value between 0 and 1. It's often used in binary classification problems where the goal is to produce a probability output.
3. **Tanh (Hyperbolic Tangent) Activation Function:**
    - The tanh function is another type of sigmoid function.
    
    $$
    f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
    $$
    
    - It maps the input to a value between -1 and 1, providing a stronger output than the sigmoid function. Tanh is commonly used in hidden layers of neural networks.
4. **Softmax Activation Function:**
    - The softmax function is often used in the output layer of a neural network for multi-class classification problems.
    - It takes a vector of arbitrary real-valued scores (logits) as input and normalizes them into a probability distribution. The output of the softmax function is a vector where each element represents the probability of a particular class.

## Unsupervised Learning of Neural Networks

![image](https://github.com/allelbhagya/nn/assets/80905783/d8a1880f-19ad-4f81-91b8-c494a0983e2c)


Unsupervised training of neural networks involves training a neural network on a dataset of unlabeled data. This means that the data does not have any associated labels or outputs. The goal of unsupervised training is to learn the underlying patterns and structure of the data without any prior knowledge.

There are a number of different unsupervised learning algorithms that can be used to train neural networks. Some common examples include:

- **Clustering:** Clustering algorithms group similar data points together. This can be useful for identifying different groups of customers, products, or other entities.
- **Dimensionality reduction:** Dimensionality reduction algorithms reduce the number of features in a dataset while preserving as much information as possible. This can be useful for improving the performance of machine learning algorithms on high-dimensional data.
- **Anomaly detection:** Anomaly detection algorithms identify data points that deviate from the expected patterns. This can be useful for detecting fraud, security breaches, and other anomalies.

To train a neural network for unsupervised learning is to use a loss function that measures the distance between the input data and the output of the network. The network is then trained to minimize this loss function.

## Restricted Boltzmann Machines

![image](https://github.com/allelbhagya/nn/assets/80905783/f993faca-48f3-4914-9375-c2a37b65a1ff)


**There are five main parts of a basic RBM:**
• Visible units
• Hidden units
• Weights
• Visible bias units
• Hidden bias units

- RBMs are a type of neural network that can learn from unlabeled data.
- RBMs have two layers of neurons: a visible layer and a hidden layer.
- The visible layer represents the input data, while the hidden layer represents features that are learned by the network.
- RBMs are trained to learn the probability distribution of the input data.
- Once trained, RBMs can be used to generate new data samples, or to infer the hidden features of a given input data sample.
- RBMs can also be used to pre-train the layers of a deep neural network, which can improve the performance of the network on supervised learning tasks.

**Training**
The technique known as pretraining using RBMs means teaching it to reconstruct the original data from a limited sample of that data. That is, given a chin, a trained network could approximate (or “reconstruct”) a face. RBMs learn to reconstruct the input dataset.

## Reconstruction 
![image](https://github.com/allelbhagya/nn/assets/80905783/1dca1fa4-00f5-4f23-ab4d-8cafea3f4df4)


In simpler terms, RBMs are a type of machine learning algorithm that can learn from data without being told what to look for. They can be used to find patterns in data, generate new data, and even pre-train other machine learning algorithms.

- **Image denoising:** RBMs can be used to remove noise from images without losing important details.
- **Image generation:** RBMs can be used to generate new images, such as realistic faces or landscapes.
- **Natural language processing:** RBMs can be used to learn the relationships between words in a language. This can be used for tasks such as machine translation and text summarization.
- **Recommendation systems:** RBMs can be used to build recommendation systems that suggest products, movies, and other items to users based on their past behavior.
- **Anomaly detection:** RBMs can be used to detect unusual patterns in data, such as fraudulent transactions or security breaches.

### Auto-Encoders

Autoencoder architecture
![image](https://github.com/allelbhagya/nn/assets/80905783/21423d81-d04a-4469-84ea-70e779afd51f)


![image](https://github.com/allelbhagya/nn/assets/80905783/9aad687c-0d30-4586-8973-d826d837e3a7)


**Autoencoders** are a type of neural network that can be used to learn efficient representations of data. They are composed of two parts: an encoder and a decoder. The encoder compresses the input data into a smaller representation, while the decoder decompresses the representation back into the original data. The autoencoder is trained to minimize the difference between the input data and the output data.

**Decoders** are the part of the autoencoder that reconstructs the input data from the compressed representation. They are typically composed of a series of neural network layers that gradually increase in size. The decoder learns to reverse the process of the encoder, and to reconstruct the input data as accurately as possible.

Autoencoders can be used for a variety of tasks, including:

- **Dimensionality reduction:** Autoencoders can be used to reduce the number of features in a dataset while preserving as much information as possible. This can be useful for improving the performance of machine learning algorithms on high-dimensional data.
- **Anomaly detection:** Autoencoders can be used to detect anomalies in data by identifying patterns that deviate from the expected behavior. This can be useful for detecting fraud, security breaches, and other problems.
- **Data denoising:** Autoencoders can be used to remove noise from data by learning the underlying patterns in the data. This can be useful for improving the quality of images, audio, and other types of data.
- **Feature learning:** Autoencoders can be used to learn new features from data. This can be useful for improving the performance of machine learning algorithms on tasks such as classification and regression.

## Defining features of autoencoders

**Autoencoders differ from multilayer perceptrons in a couple of ways:**
• They use unlabeled data in unsupervised learning.
• They build a compressed representation of the input data.

**Unsupervised learning of unlabeled data.** The autoencoder learns directly from unlabeled data. This is connected to the second major difference between multilayer perceptrons and autoencoders.

**Learning to reproduce the input data.** The goal of a multilayer perceptron network is to generate predictions over a class (e.g., fraud versus not fraud). An autoencoder is trained to reproduce its own input data.

**Training autoencoders** Autoencoders rely on backpropagation to update their weights. The main difference between RBMs and the more general class of autoencoders is in how they calculate the gradients.
