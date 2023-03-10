# Week 1: Neural Networks
## Neural networks intuition
### Neurons and the brain
* _Neural Networks_ has started with the motivation of trying to build software to mimic the brain.
* Since around 2005, it has been rebranded into _Deep Learning_.
* Currently they are used in many fields like:
    * Speech Recognition
    * Computer Vision
    * Natural Language Processing

![Biological vs Artificial Neuron](./images/neural-network-01.jpg)
* The artificial neural network uses a very simplified Mathematical model of what a biological neuron does
    * What a neuron does is it takes some inputs, one or more inputs, which are just numbers. 
    * It does some computation and it outputs some other number, which then could be an input to a second neuron.
    * When you're building an artificial neural network or deep learning algorithm, rather than building one neuron at a time, you often want to simulate many such neurons at the same time.

### Example: Demand Prediction
* We'll use an example from demand prediction in which you look at the product and try to predict, will this product be a top seller or not?
* In this example, you're selling T-shirts and you would like to know if a particular T-shirt will be a top seller, yes or no, and you have collected data of different t-shirts that were sold at different prices, as well as which ones became a top seller.
* In this example, the input feature $x = \text{price}$ of the T-shirt, and so that's the input to the learning algorithm. 
* If you apply _logistic regression_ to fit a _sigmoid function_ to the data that might look like that then the outputs of your prediction might look like this $f(x) = \frac{1}{1+e^{-(wx + b)}}$
![Demand Prediction using Logistic regression](./images/neural-network-02.jpg)
* To set us up to build a neural network we are going to switch the terminology and use alphabet $a$ to denote the _logistic regression_ output.
    * The term $a$ stands for _activation_
* This logistic regression units or this little logistic regression algorithm, can be thought of as a very simplified model of a single neuron in the brain.
    * Where what the neuron does is it takes us input the price $x$, and then it computes this formula on top, and it outputs the number $a$, which is computed by this formula, and it outputs the probability of this T-shirt being a top seller.
* Given this description of a single neuron, building a neural network now it just requires taking a bunch of these neurons and wiring them together or putting them together.
----------
* Let's now look at a more complex example of demand prediction.
![Complex Demand Prediction](./images/neural-network-03.jpg)
* In this example, we're going to have four features to predict whether or not a T-shirt is a top seller.The features are:
    * Price.
    * Shipping Cost.
    * Amount of Marketing.
    * Material Quality.
* you might suspect that whether or not a T-shirt becomes a top seller actually depends on a few factors:
    * Affordability
    * Awareness
    * Perceived Quality
* What I'm going to do is:
    * Create one artificial neuron to try to estimate the probability that this T-shirt is perceive as highly affordable. 
        * _Affordability_ is mainly a function of _price_ and _shipping costs_. 
        * We're going to use a little neuron here, a logistic regression unit to input _price_ and _shipping costs_ and predict do people think this is affordable? 
    * Second, I'm going to create another artificial neuron here to estimate, is there high awareness of this? 
        * _Awareness_ in this case is mainly a function of the _marketing_ of the T-shirt. 
    * Finally, going to create another neuron to estimate do people perceive this to be of _high quality_
        * That may mainly be a function of the _price_ of the T-shirt and of the _material quality_. 
        * _Price_ is a factor here because if there's a very high priced T-shirt, people will sometimes perceive that to be of high quality.
* Given these estimates of _affordability_, _awareness_, and _perceived quality_ we then wire the outputs of these three neurons to another neuron, that then there's another logistic regression unit That finally inputs those three numbers and outputs the probability of this t-shirt being a top seller.
* In the terminology of neural networks, we're going to group these three neurons together into what's called a _layer_. 
    > **Terminology** _Layer_:  
    > A _layer_ is a grouping of neurons which takes as input the same or similar features, and that in turn outputs a few numbers together.  
    > A _layer_ can have multiple neurons or it can also have a single neuron:
    > * The _input layer_: is the layer of the input features. $\vec{x}$
    > * The _hidden layer_: is the layer(s) in the middle of the network that calculate the activation values. $\vec{a}$
    > * The _output layer_: is the layer with the final neuron that the output probability predicted by the neural network.

* In the terminology of neural networks we're also going to call affordability, awareness and perceived quality to be _activations_.

* In this example, this particular neural network therefore carries out computations as follows:
    1. It inputs 4 numbers.
    2. Then this layer of the neural network uses those 4 numbers to compute the new numbers also called _activation values_ (3 numbers).
    3. Then the final layer, the output layer of the neural network used those 3 numbers to compute 1 number.
* In a neural network this list of 4 numbers is also called the _input layer_.

> The way a neural network is implemented in practice each neuron in a certain layer will have access to every feature from the previous layer.   
You can imagine that if you're trying to predict affordability and it knows what's the price shipping cost marketing and material, may be you'll learn to ignore marketing and material and just figure out through setting the parameters appropriately to only focus on the subset of features that are most relevant to affordability.

### Example: Recognizing Images
* If you're building a face recognition application, you might want to train a neural network that takes as input a picture like this and outputs the identity of the person in the picture.
![Recognizing Images](./images/neural-network-04.jpg)
* This image is 1,000 by 1,000 pixels. Its representation in the computer is actually as 1,000 by 1,000 grid, or also called 1,000 by 1,000 matrix of _pixel intensity values_.
* If you were to take these pixel intensity values and unroll them into a vector, you end up with a list or a vector of a million pixel intensity values.  
    >The face recognition problem is, can you train a neural network that takes as input a feature vector with a million pixel brightness values and outputs the identity of the person in the picture.  

    ![Face Recognition](./images/neural-network-05.jpg)
* This is how you might build a neural network to carry out this task. The input image $\vec{\mathbf{x}}$ is fed to this layer of neurons. 
    * This is the first hidden layer, which then extract some features. 
    * The output of this first hidden layer is fed to a second hidden layer
    * That output is fed to a third layer
    * Then finally to the output layer, which then estimates, say the probability of this being a particular person.
----------
## Neural network model
### Neural network layer
* The fundamental building block of most modern neural networks is a layer of neurons.
* When you build neural networks with multiple layers, it'll be useful to give the layers different numbers.
* By convention, we give numbers to our layers:
    * _Layer 0_ (Input $\vec{\mathbf{x}}$)
    * _Layer 1_
    * _Layer 2_
    * ... etc
* In order to introduce notation to help us distinguish between the different layers, we are going to add superscript index to the output and variables of each layer:
    * $\vec{a}^{[1]}$ is the output of layer 1
    * The activation function of _neuron 1_ in _layer 1_ is:

$$
\vec{a}^{[1]} = g(\vec{\mathbf{w}_{1}}^{[1]} \cdot \vec{\mathbf{x}} + b_{1}^{[1]})
$$

### More complex neural networks
![More complex neural networks](./images/neural-network-06.jpg)

The general notation form of this equation for an arbitray layer $l$ and for an arbitray unit $j$ is:

$$
a_{j}^{[l]} = g(\vec{\mathbf{w}}_{j}^{[l]} \cdot \vec{a}^{[l-1]} + b_{j}^{[l]})
$$

> **Note**  
> This form is generic so when we apply it to the input later _Layer 0_ it becomes:  
> $$\vec{\mathbf{x}} = \vec{a}^{[0]} $$

### Inference: making predictions (forward propagation)
![Forward Propagation](./images/neural-network-07.jpg)
* This computation goes from left to right, you start from $\vec{\mathbf{x}}$ and compute $\vec{a}^{[1]}$, then $\vec{a}^{[2]}$, then $\vec{a}^{[3]}$.
* This algorithm is also called _forward propagation_ because you're propagating the activations of the neurons.
* This is in contrast to a different algorithm called _backward propagation_ or _back propagation_, which is used for learning.
* This type of neural network architecture where you have more hidden units initially and then the number of hidden units decreases as you get closer to the output layer, is a pretty typical choice when choosing neural network architectures.

[Lab: Neurons and Layers](./code/C2_W1_Lab01_Neurons_and_Layers.ipynb)

----------

## TensorFlow implementation
* [TensorFlow](https://www.tensorflow.org/) is one of the leading frameworks to implementing deep learning algorithms. 
* The other popular tool is [PyTorch](https://pytorch.org/). 
* We're going to focus in this specialization on TensorFlow.
### Inference in Code
To build a layer in TensorFlow, we do it like the below example:
```python
# First we define NumPy Array for the data labels
x = np.array([[200.0, 17.0]])
# Then we define the layer using Dense layer
# There are other layer types
layer_1 = Dense(units=3, activation='sigmoid')
# Finally we get the activation of layer_1
a1 = layer_1(x)
```

For _Layer 2_ in the neural network:
```python
layer_2 = Dense(units=1, activation='sigmoid')
a1 = layer_1(a1)
```
### Data in TensorFlow
* Data is represented differently in NumPy and TensorFlow.
* TensorFlow is using matrix notation (instead of 1D vectors like in [previous course](../../Course-01-Supervised%20Machine%20Learning/README.md)) to effeciently handle large datasets.
#### NumPy matrix 
* **2D Matrix** example is $2 \times 3$ matrix (2 rows and 3 columns) and is represented in NumPy as:

$$\begin{bmatrix} 1 & 2 & 3 \newline 4 & 5 & 6 \end{bmatrix}$$

```python
x = np.array([[1, 2, 3], 
              [4, 5, 6]])
```
* **2D Matrix** example is $4 \times 2$ matrix (4 rows and 2 columns) and is represented in NumPy as:

$$\begin{bmatrix} 0.1 & 0.2 \newline -3 & -4 \newline -0.5 & -0.6 \newline 7 & 8 \end{bmatrix}$$

```python
x = np.array([[0.1, 0.2],
              [-3.0, -4.0], 
              [-0.5, -0.6],
              [7.0, 8.0]])
```
* **2D Matrix** example is $1 \times 2$ matrix (also called _row vector_) and is represented in NumPy as:

$$\begin{bmatrix}200 & 17\end{bmatrix}$$

```python
x = np.array([[200, 17]])
```
* **2D Matrix** example is $2 \times 1$ matrix (also called _column vector_) and is represented in NumPy as:

$$\begin{bmatrix}200 \newline 17\end{bmatrix}$$

```python
x = np.array([[200], 
              [17]])
```
* **1D Vector** is normal array and represented in NumPy as (notice the single brackets insead of double brackets like previous examples):
    ```python
    x = np.array([200, 17])
    ```
#### TensorFlow representation
* While TensorFlow accepts data in NumPy, the data returned from TensorFlow layers (i.e. activation vectors) as object of type `tf.Tensor`:
    ```
    tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
    ```
* We can convert `tf.Tensor` into NumPy vector:
    ```python
    a1.numpy()
    ```
### Building a neural network
* We can build our neural network in easier way using `Sequential` 
    ```python
    layer_1 = Dense(units=3, activation='sigmoid')
    layer_2 = Dense(units=1, activation='sigmoid')
    model = Sequential([layer_1, layer_2])
    ```
* Also by convention, it is possible to simplify the above code:
    ```python
    model = Sequential([
        Dense(units=3, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')])
    ```
* To train our model we call methods `compile` and `fit`:
    ```python
    model.compile(...)
    model.fit(x, y)
    ```
* To use our model to predict, we call method `predict`:
    ```python
    model.predict(x_new)
    ```

[Lab: Coffee Roasting in Tensorflow](./code/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)

----------
## Neural network implementation in Python
[Lab: CoffeeRoastingNumPy](./code/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

----------
## Speculations on artificial general intelligence (AGI)
**AI** (_Aritificial Intelligence_) include 2 branches:
1. **ANI**: Aritificial Narrow Intelligence
    * E.g., smart speakers, self-driving car, web search, AI in farming and factories.
    * This field has recently seen great breakthroughs
2. **AGI** Artificial General Intelligence
    * Do anything a human can do.
    * This field didn't see a lot of progress so far.
----------
## Vectorization
### Matrix multiplication
#### Dot products 
* Consider 2 vectors $\vec{a}$ and $\vec{\mathbf{w}}$

$$
\vec{a} = \begin{bmatrix}1 \newline 2 \end{bmatrix} \qquad \vec{\mathbf{w}} = \begin{bmatrix}3 \newline 4 \end{bmatrix}
$$

* So the _dot product_ of $\vec{a} \cdot \vec{\mathbf{w}}$ is: 

$$
\begin{align*}
z &= \vec{a} \cdot \vec{\mathbf{w}} \newline
& = \begin{bmatrix}1 \newline 2 \end{bmatrix} \cdot \begin{bmatrix}3 \newline 4 \end{bmatrix} \newline
& = (1 \times 3) + (2 \times 4) \newline
& = 11
\end{align*}
$$

* There is an equivalent way to write a _dot product_ by using the transpose of one of the vectors (convert a column vector into a row vector) and use vector vector multiplication:

$$
\vec{\mathbf{a}} = \begin{bmatrix}1 \newline 2 \end{bmatrix} \rightarrow \vec{\mathbf{a}}^\intercal = \begin{bmatrix}1 & 2 \end{bmatrix} \newline
$$

$$
\mathbf{z} = \vec{a}^\intercal \vec{\mathbf{w}}
$$

#### Vector Matrix Multiplication
* Consider $\vec{a}$ and $\mathbf{W}$:

$$
\vec{a} = \begin{bmatrix} 1 \newline 2 \end{bmatrix} \quad \text{Then: } \quad \vec{a}^\intercal = \begin{bmatrix} 1 & 2 \end{bmatrix} \newline
\mathbf{W} = \begin{bmatrix}3 & 5 \newline 4 & 6\end{bmatrix}
$$

* So $\mathbf{Z}$ will be $1 \times 2$ matrix:

$$
\begin{align*}
\mathbf{Z} & = \vec{a}^\intercal \mathbf{W} \\
& = \begin{bmatrix} \vec{a}^\intercal \vec{\mathbf{w_{1}}} & \vec{a}^\intercal \vec{\mathbf{w_{2}}} \end{bmatrix} \newline
& = \begin{bmatrix} (1 \times 3)+(2 \times 4) & (1 \times 5)+(2 \times 6) \end{bmatrix} \newline
& = \begin{bmatrix} 11 & 17 \end{bmatrix}
\end{align*}
$$

#### Matrix Matrix Multiplication
* First, we need to take transpose of a matrix
    * We take each column and transpose it into a row

$$
\mathbf{A} = \begin{bmatrix}1 & -1 \newline 2 & -2\end{bmatrix} \quad \rightarrow \quad \mathbf{A}^\intercal = \begin{bmatrix}1 & 2 \newline -1 & -2\end{bmatrix}
$$

* In ML context, we think of matrix as set of columns and transposed matrix as set of rows
* To calculate matrix matrix multiplication $\mathbf{Z}$

$$
\begin{align*}
\mathbf{Z} & = \mathbf{A}^\intercal \mathbf{W} \\
& = \begin{bmatrix}\leftarrow & \vec{\mathbf{a}}_{1}^\intercal & \rightarrow \newline \leftarrow & \vec{\mathbf{a}}_{2}^\intercal & \rightarrow \end{bmatrix} \begin{bmatrix}\uparrow & \uparrow \newline \vec{\mathbf{w}}_{1} & \vec{\mathbf{w}}_{2} \newline \downarrow & \downarrow\end{bmatrix} \newline
& = \begin{bmatrix}\vec{\mathbf{a}}_{1}^\intercal \vec{\mathbf{w}}_{1} & \vec{\mathbf{a}}_{1}^\intercal \vec{\mathbf{w}}_{2} \newline \vec{\mathbf{a}}_{2}^\intercal \vec{\mathbf{w}}_{1} & \vec{\mathbf{a}}_{2}^\intercal \vec{\mathbf{w}}_{2}\end{bmatrix} \newline
& = \begin{bmatrix}11 & 17 \newline -11 & -17\end{bmatrix}
\end{align*}
$$

### Matrix multiplication rules
* We can think of vector $\vec{\mathbf{a}}$ as _column vector_ while $\vec{\mathbf{a}}^\intercal$ as _row vector_.

$$
\vec{\mathbf{a}} = \begin{bmatrix}1 \newline 2 \newline 3\end{bmatrix} \quad \rightarrow \quad \vec{\mathbf{a}}^\intercal = \begin{bmatrix}1 & 2 & 3\end{bmatrix}
$$

* We can think of a matrix as set of _column vectors_ and to transpose a matrix we transpose each of its _column vectors_ into _row vectors_

$$
\mathbf{A} = \begin{bmatrix}\uparrow & \uparrow & \uparrow \newline \vec{\mathbf{a_{1}}} & \vec{\mathbf{a_{2}}} & \vec{\mathbf{a_{3}}} \newline \downarrow & \downarrow & \downarrow\end{bmatrix} \quad \rightarrow \quad \mathbf{A}^\intercal = \begin{bmatrix} \leftarrow & \vec{\mathbf{a_{1}}}^\intercal & \rightarrow \newline \leftarrow & \vec{\mathbf{a_{2}}}^\intercal & \rightarrow \newline \leftarrow & \vec{\mathbf{a_{3}}}^\intercal & \rightarrow \end{bmatrix}
$$

* To calculate $\mathbf{Z}$ as the multiplication of $\mathbf{A}^\intercal$ by $\mathbf{W}$:

$$
\mathbf{A}^\intercal = \begin{bmatrix} \leftarrow & \vec{\mathbf{a_{1}}}^\intercal & \rightarrow \newline \leftarrow & \vec{\mathbf{a_{2}}}^\intercal & \rightarrow \newline \leftarrow & \vec{\mathbf{a_{3}}}^\intercal & \rightarrow \end{bmatrix} \quad  \mathbf{W} = \begin{bmatrix}\uparrow & \uparrow & \uparrow & \uparrow \newline \vec{\mathbf{w_{1}}} & \vec{\mathbf{w_{2}}} & \vec{\mathbf{w_{3}}} & \vec{\mathbf{w_{4}}} \newline \downarrow & \downarrow & \downarrow & \downarrow \end{bmatrix}
$$

$$
\begin{align*}
\mathbf{Z} & = \mathbf{A}^\intercal \mathbf{W} \newline
& = \begin{bmatrix}(\vec{\mathbf{a}}_{1}^\intercal \cdot \vec{\mathbf{w}}_{1}) & (\vec{\mathbf{a}}_{1}^\intercal \cdot \vec{\mathbf{w}}_{2}) & (\vec{\mathbf{a}}_{1}^\intercal \cdot \vec{\mathbf{w}}_{3}) & (\vec{\mathbf{a}}_{1}^\intercal \cdot \vec{\mathbf{w}}_{4}) \newline (\vec{\mathbf{a}}_{2}^\intercal \cdot \vec{\mathbf{w}}_{1}) & (\vec{\mathbf{a}}_{2}^\intercal \cdot \vec{\mathbf{w}}_{2}) & (\vec{\mathbf{a}}_{2}^\intercal \cdot \vec{\mathbf{w}}_{3}) & (\vec{\mathbf{a}}_{2}^\intercal \cdot \vec{\mathbf{w}}_{4}) \newline (\vec{\mathbf{a}}_{3}^\intercal \cdot \vec{\mathbf{w}}_{1}) & (\vec{\mathbf{a}}_{3}^\intercal \cdot \vec{\mathbf{w}}_{2}) & (\vec{\mathbf{a}}_{3}^\intercal \cdot \vec{\mathbf{w}}_{3}) & (\vec{\mathbf{a}}_{3}^\intercal \cdot \vec{\mathbf{w}}_{4})\end{bmatrix}
\end{align*}
$$

> **Note**: 
> * Matrix multiplication is only valid if the number columns of the $1^{\text{st}}$ matrix is equal to the number of rows of the $2^{\text{nd}}$ matrix.
> * The result will be a matrix with dimensions # rows of $1^{\text{st}}$ matrix $\times$ # columns of $2^{\text{nd}}$ matrix.

[Practice Lab: Neural Networks for Binary Classification](./assignment/C2_W1_Assignment.ipynb)

[<<Previous](../README.md) | [Next>>](../week-02/README.md)