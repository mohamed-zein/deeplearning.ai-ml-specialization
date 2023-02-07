# Neural network training
## Neural Network Training
### TensorFlow implementation
![Neural Network](./images/neural-network-01.PNG)
Given set of $(x,y)$ examples, how to build and train neural network in the code?
1. Ask TensorFlow to string together the neural network layers:
    ```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential(
        [               
            tf.keras.Input(shape=(400,)),
            Dense(25, activation='sigmoid', name = 'layer1'),
            Dense(15, activation='sigmoid', name = 'layer2'),
            Dense(1, activation='sigmoid', name = 'layer3')
        ], name = "my_model" 
    )
    ```
2. Ask TensorFlow to compile together the model and specify the _Loss_ function (in this case we use `BinaryCrossentropy` loss function)
    ```python
    from tensorflow.keras.losses import BinaryCrossentropy

    model.compile(loss=BinaryCrossentropy())
    ```
3. Call the `fit` function to fit the model to the data set `X, Y` and specify how many steps in the gradient descent `epochs`
    ```python
    model.fit(X, y, epochs=100)
    ```
### Training Details
#### Model Training Steps
<table>
<tr>
<th>Step</th><th>Logistic Regression</th><th>Neural Network</th>
</tr>
<tr>
<td>

Speicfy how to compute output given input $x$ and parameters $w, b$ (defined model)

</td>
<td>

```python
z = np.dot(w,x) + b
f_x = 1/(1+np.exp(-z))
```

</td>
<td>

```python
model = Sequential([ 
            Dense(...),
            Dense(...),
            Dense(...)
        ])
```

</td>
</tr>
<tr>
<td>

Specify _loss_ and _cost_<br>$L(f_{\vec{\mathbf{w}},b}(\vec{\mathbf{X}}), \mathbf{y})$ <br> $J(\vec{\mathbf{w}}, b) = \frac{1}{m} \sum\limits_{i=i}^{m}{L(f_{\vec{\mathbf{w}},b}(\vec{\mathbf{X}}^{(i)}), \mathbf{y}^{(i)})}$

</td>
<td>

```python
loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x)
```

</td>
<td>

```python
model.compile(loss=BinaryCrossentropy())
```

</td>
</tr>
<tr>
<td>

Train on data to minimize $J(\vec{\mathbf{w}}, b)$ using Gradient Descent

</td>
<td>

```python
w = w - alpha * dj_dw
b = b - alpha * dj_db
```

</td>
<td>

```python
model.fit(X, y, epochs=100)
```

</td>
</tr>
</table>

1. In the first step we create the model and define the layers and the activation function of each layer.
2. Loss and Cost functions
    * `BinaryCrossentropy` is commonly used with Logistic Binary classification.
    * `MeanSquaredError` is commonly used with regression.
3. Gradient Descent
    * TensorFlow uses **back propagation** to compute derivatives used in gradient descent.

## Activation Functions
### Alternatives to the sigmoid activation
![Examples of Activation Functions](./images/activation-function-01.jpg)
* We have already seen the **sigmoid** activation function $g(\mathbf{z}) = \frac{1}{1+e^{-\mathbf{z}}}$
* Another common activation function is **ReLU** (Rectified Linear Unit) $g(\mathbf{z}) = \max(0,\mathbf{z})$
* Also, we have the **Linear Activation Function** $g(\mathbf{z}) = \mathbf{z}$
### Choosing activation functions
#### Output Layer
* It turns out that there'll often be one fairly natural choice, depending on what is the target or the ground truth label $\mathbf{y}$.
    * When working on a binary classification problem where $\mathbf{y}$ is either $0$ or $1$, then the **sigmoid** activation function will almost always be the most natural choice.
    * When working on regression problem where $\mathbf{y} = +/-$, we can use the **Linear Activation Function**.
    * When working on regression problem where $\mathbf{y} = 0 \text{ or } +$ then we can use **ReLU**
#### Hidden Layer
* **ReLU** activation function is the most commonly used activation function used.
    * It is much faster that the sigmoid
### Why do we need activation functions?
* If we used **Linear Activation Functino** in all the neurons in the neural network, then the whole neural network will act as simple Linear Regression.

[Lab: ReLU activation](./code/C2_W2_Relu.ipynb)

[<<Previous](../week-01/README.md) | [Next>>]()