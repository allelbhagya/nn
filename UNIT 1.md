# Neural Networks

# UNIT-I Introduction to Neural Networks

**UNIT-I Introduction to Neural Networks: Neural Network, Human Brain, Models of Neuron, Neural networks viewed as directed graphs, Biological Neural Network, Artificial neuron, Artificial Neural Network architecture, Artificial Intelligence and Neural Networks; Network Architectures, Single-layered Feed forward Networks, Multi-layered Feed forward Networks, Recurrent Networks, Topologies**

## Neural Network

Neural networks are a computational model that shares some properties with the animal brain in which many simple units work in parallel with no centralized control unit. The weights between units are the primary means of long-term information storage, and updating these weights is the primary way neural networks learn new information.

A network’s architecture can be defined by:

- Number of neurons
- Number of layers
- Types of connections between layers

![image](https://github.com/allelbhagya/nn/assets/80905783/17851b2f-00e9-47be-a86e-2b679035525b)


## Biological Neural Network

### Synapses

- Connecting junctions between axons and dendrites.
- Synapses send signals from the axon of a neuron to the dendrite of another neuron.

### Dendrites

- Fibers branching out from the soma in a bushy network around the nerve cell.
- Allow the cell to receive signals from connected neighboring neurons.
- Each dendrite performs multiplication by its weight value.

### Axons

- Single, long fibers extending from the main soma.
- Stretch out longer distances than dendrites.
- Neurons send electrochemical pulses through axons, generating action potential.

### Information Flow Across the Biological Neuron

- Excitatory synapses increase the potential; inhibitory synapses decrease it.
- Plasticity refers to long-term changes in connection strength in response to input stimulus.
- Neurons can form new connections over time.

### From Biological to Artificial

- Studying the basic components of the brain allows for the understanding of the fundamental components of the mind.
- Research has shown ways to map out brain functionality and track signals as they move through neurons.

  ![1_SgJPlfKHwL-qis-aL19Fyg](https://github.com/allelbhagya/nn/assets/80905783/78b3ea4a-158f-4bd5-af8f-0edbd4946c96)

## Models of Neurons

### 1. Perceptron

![image](https://github.com/allelbhagya/nn/assets/80905783/017dcc02-2b7c-4a20-8d9d-c1728836dec6)


- Foundational unit of artificial neural networks.
- Makes binary decisions by summing weighted inputs and passing the result through an activation function.
- Adjusts weights during training based on decision correctness.

### 2. ReLU (Rectified Linear Unit)

![image](https://github.com/allelbhagya/nn/assets/80905783/d17b8cf0-b450-4cb8-b402-e8d8caae69db)


- Activation function introducing non-linearity.
- Outputs the input for positive values and zero for negative values.
- Effective in mitigating the vanishing gradient problem.

### 3. Leaky ReLU

![image](https://github.com/allelbhagya/nn/assets/80905783/95751831-e958-41c6-a58d-e83c30913316)


- Variant of ReLU allowing a small, non-zero gradient for negative inputs.
- Prevents neurons from becoming entirely inactive.

### 4. LSTM (Long Short-Term Memory)

![image](https://github.com/allelbhagya/nn/assets/80905783/399929e7-11f8-40cf-846e-c6a510cb8545)


- Recurrent neural network (RNN) architecture.
- Addresses the vanishing gradient problem and captures long-term dependencies in sequential data.
- Incorporates memory cells with input, output, and forget gates.

### 5. Sigmoid

![image](https://github.com/allelbhagya/nn/assets/80905783/e06c23aa-cb1b-49a3-8e70-cc7f9fce87f5)


- Activation function mapping input values to an output range between 0 and 1.
- Commonly used in binary classification problems.
- Suitable for modeling probabilities.

# Neural Network as Directed Graph

![1_JOvnkqDOkJxJHwM0M7F_ZQ](https://github.com/allelbhagya/nn/assets/80905783/2d9f58bf-fbe9-42c3-979e-cb81471b92fd)

## Definition:

- **Directed Graph:**
    - A directed graph, or digraph, consists of nodes (vertices) and directed edges (arcs) connecting these nodes.
    - In a neural network, nodes represent artificial neurons, and directed edges represent the connections between neurons.
    - The direction of the edges signifies the flow of information, typically from input layers to output layers.

## Representation:

- **Nodes (Vertices):**
    - Each node in the graph corresponds to an artificial neuron. These neurons are organized into layers: input layer, hidden layers, and output layer.
    - The input layer nodes receive external input signals, and the output layer nodes produce the final output of the network.
- **Directed Edges (Arcs):**
    - Directed edges connect the neurons, indicating the flow of information from one neuron to another.
    - The edges are associated with weights, reflecting the strength of the connection between neurons.
    - The direction of the edge signifies the flow of information—usually from the input layer towards the output layer.

# Artificial Neuron

![The-structure-of-the-artificial-neuron](https://github.com/allelbhagya/nn/assets/80905783/4fb2478d-3edd-4ea7-af07-33ffae3ff713)


An artificial neuron, often referred to as a perceptron, serves as the fundamental building block in artificial neural networks, inspired by the structure and function of biological neurons in the human brain. 

## Components:

1. **Inputs:**
    - An artificial neuron receives input signals, which represent features or characteristics of the input data.
2. **Weights:**
    - Each input is associated with a weight, indicating its importance or contribution to the neuron's output.
    - Weights are adjustable parameters that the neuron learns during the training process.
3. **Summation Function:**
    - The neuron computes a weighted sum of its inputs. This involves multiplying each input by its corresponding weight and summing up these products.
    - Mathematically, the summation function is represented as the dot product of the input vector and weight vector:
    - $\text{Sum} = \sum_{i=1}^{n} \text{input}_i \times \text{weight}_i$
4. **Activation Function:**
    - The result of the summation is passed through an activation function. This function determines whether the neuron should be activated (output a signal) based on the computed sum.
    - Common activation functions include the step function (for binary output), sigmoid, hyperbolic tangent (tanh), ReLU, and others.
    - $\text{Output} = \text{Activation}(\text{Sum})$
5. **Bias (Optional):**
    - In addition to weights, a neuron may have a bias term. The bias is a constant value that helps the neuron adjust its decision boundary.
    - The bias is added to the weighted sum:
    - $\text{Sum} = \sum_{i=1}^{n} \text{input}_i \times \text{weight}_i + \text{bias}$
6. **Output:**
    - The final output of the neuron is the result of the activation function. This output is then used as input for subsequent layers in a neural network.

## Functionality:

- **Learning:**
    - During the training phase, the neuron adjusts its weights (and bias, if applicable) based on the correctness of its output concerning the expected output.
    - Learning enables the neuron to make more accurate decisions over time.
- **Organization:**
    - Artificial neurons are organized into layers to form neural networks.
    - The collective behavior of interconnected neurons allows neural networks to model complex relationships and make predictions or classifications based on input data.
    
    # Artificial Neural Network Architecture

  ![image](https://github.com/allelbhagya/nn/assets/80905783/f7217651-2e2a-49ff-b1a3-5436dd1193cd)

    
    The architecture of an artificial neural network (ANN) refers to its structural organization, including the arrangement and connectivity of neurons. Here's an overview of the key components and concepts related to ANN architecture:
    
    ### 1. **Neurons:**
    
    - Neurons are the basic processing units in a neural network. Each neuron receives input, processes it, and produces an output.
    
    ### 2. **Layers:**
    
    - Neural networks are organized into layers, each serving a specific purpose.
    - **Input Layer:** The first layer that receives external input.
    - **Hidden Layers:** Intermediate layers between the input and output layers. They contribute to the network's ability to learn complex representations.
    - **Output Layer:** The final layer that produces the network's output.
    
    ### 3. **Connections (Edges):**
    
    - Connections between neurons, represented by weights, determine the strength of the influence one neuron has on another.
    
    ### 4. **Weights and Biases:**
    
    - Weights are associated with connections between neurons and determine the strength of those connections.
    - Biases are optional constants that provide additional flexibility in adjusting the decision boundaries of neurons.
    
    ## Single-layered Feedforward Networks:

  ![image](https://github.com/allelbhagya/nn/assets/80905783/98ae6bb1-0a49-48da-bd2e-8c5822d51072)

    
    ### Definition:
    
    A single-layered feedforward network, also known as a perceptron, consists of input nodes connected directly to an output node. It processes input data in a straightforward manner, making decisions or predictions based on weighted inputs.
    
    ### Characteristics:
    
    1. **No Hidden Layers:**
        - Only contains an input layer and an output layer.
    2. **Binary Output:**
        - Commonly used for binary classification tasks.
    3. **Linear Decision Boundary:**
        - Limited to linear decision boundaries, making it suitable for linearly separable problems.
    4. **Limitations:**
        - Limited capacity to learn complex patterns, suitable for simple tasks.
    
    ## Multi-layered Feedforward Networks:

![image](https://github.com/allelbhagya/nn/assets/80905783/885e74a8-6df9-44f7-b606-8c6d649f739a)

## Definition:
    
Multi-layered feedforward networks, or multilayer perceptrons (MLPs), consist of an input layer, one or more hidden layers, and an output layer. Information flows through the network without cycles, making it a feedforward architecture.
    
### Characteristics:
    
1. **Hidden Layers:**
        - Intermediate layers process and transform data, allowing the network to learn complex representations.
2. **Non-linear Activation:**
        - Utilizes non-linear activation functions (e.g., ReLU, sigmoid) to introduce non-linearity.
3. **Backpropagation:**
        - Trained using backpropagation, adjusting weights to minimize errors during training.
4. **Versatility:**
        - Suitable for a wide range of tasks, including image recognition, natural language processing, and regression.
    
## Recurrent Networks:

![image](https://github.com/allelbhagya/nn/assets/80905783/a097d9de-a095-45cd-b0f7-6277b2d7bbf0)

### Definition:
    
Recurrent Neural Networks (RNNs) have connections that form cycles, allowing the network to maintain and use information over time. This architecture is well-suited for sequential data and tasks involving temporal dependencies.
    
### Characteristics:
    
1. **Temporal Connections:**
        - Neurons have connections that loop back, enabling the network to capture sequential patterns.
2. **Variable Input Length:**
        - Can handle input sequences of varying lengths.
3. **Memory Mechanism:**
        - Incorporates memory cells to store and retrieve information over extended sequences.
4. **Applications:**
        - Suitable for tasks like speech recognition, language modeling, and time series prediction.
    
## Topologies:
    
### Definition:
    
Network topology refers to the arrangement of nodes and connections in a neural network. Different topologies offer advantages in specific tasks and learning paradigms.
    
### Common Topologies:
    
1. **Fully Connected (Dense):**
        - All neurons in one layer are connected to every neuron in the next layer.
2. **Convolutional:**
        - Commonly used for image processing, with shared weights in convolutional layers.
3. **Recurrent:**
        - Contains cycles, allowing information to persist over time.
4. **Radial Basis Function (RBF):**
        - Employs radial basis functions as activation functions, often used for function approximation.
    
