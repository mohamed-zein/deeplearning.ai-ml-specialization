# Week 2: Regression with multiple input variables
## Multiple features
Consider the housing problem with more than one features.
Size in Feets <br> $\mathbf{x}_{1}$ | Number of bedrooms <br> $\mathbf{x}_{2}$ | Number of floors <br> $\mathbf{x}_{3}$ | Age of home in years <br> $\mathbf{x}_{4}$ | Price (\$) in \$1000's
--------------|--------------------|------------------|----------------------|-----------------------
2014 | 5 | 1 | 45 | 460
1416 | 3 | 2 | 40 | 232
1534 | 3 | 2 | 30 | 315
852 | 2 | 1 | 36 | 178

* Notations

$$
\begin{split}
\mathbf{x}_{j} & = j^{th} \text{ feature} \newline
n & = \text{number of features} \newline
\vec{\mathbf{x}}^{(i)} & = \text{features vector of } i^{th} \text{ training data} \quad  \xrightarrow{\text{example}} \quad \vec{\mathbf{x}}^{(2)} = \begin{bmatrix} 1416 & 3 & 2 & 40 \end{bmatrix} \newline
\mathbf{x}^{(i)}_{j} & = \text{Value of feature } j \text{ in } i^{th} \text{ training example} \quad  \xrightarrow{\text{example}} \quad \mathbf{x}^{(2)}_{3} = 2
\end{split}
$$

### Multiple Linear regression model using $n$ features  
* Parameters of the model are:

$$
\begin{split}
\vec{\mathbf{w}} & = \begin{bmatrix} w_{1} & w_{2} & w_{3} & \cdots & w_{n} \end{bmatrix} \newline
b & \text{ is a number} \newline
\vec{\mathbf{x}} & = \begin{bmatrix} \mathbf{x}_{1} & \mathbf{x}_{2} & \mathbf{x}_{3} & \cdots & \mathbf{x}_{n} \end{bmatrix} \newline
\end{split}
$$

* So the **Multiple Linear regression model** would be:

$$
\begin{split}
f_{\vec{w},b}(\vec{\mathbf{x}}) & = \vec{\mathbf{w}} \underbrace{\cdot}_{\text{dot product}} \vec{\mathbf{x}} + b \newline
& = w_{1}\mathbf{x}_{1} + w_{2}\mathbf{x}_{2} + w_{3}\mathbf{x}_{3} + \cdots + w_{n}\mathbf{x}_{n} + b
\end{split}
$$

## Vectorization
When using Vectorization when writing your machine learning code, it makes your program shorter and also run more effeciently.
### Vectorization example
$$
\text{Parameters and features:} \newline
\vec{\mathbf{w}} = \begin{bmatrix} w_{1} & w_{2} & w_{3} \end{bmatrix} \newline
b \text{ is a number} \newline
\vec{\mathbf{x}} = \begin{bmatrix} \mathbf{x}_{1} & \mathbf{x}_{2} & \mathbf{x}_{3} \end{bmatrix} \newline
n = 3
$$

In Python we are going to define vectors $\vec{\mathbf{w}}$ and $\vec{\mathbf{x}}$ as Arrays of Linear Algebra library [NumPy](https://numpy.org/):
```python
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10,20,30])
```

#### Without Vectorization 
We can implement the model (for small $n$) as:  

$$
f_{\vec{w},b}(\vec{\mathbf{x}}) = \mathbf{w}_{1}\mathbf{x}_{1} + \mathbf{w}_{2}\mathbf{x}_{2} + \mathbf{w}_{3}\mathbf{x}_{3} + b
$$

```python
f = w[0] * x[0] + 
    w[1] * x[1] + 
    w[2] * x[2] + b
```

For Large $n$, we can implemet:

$$
f_{\vec{w},b}(\vec{\mathbf{x}}) = \left( \sum\limits_{j=1}^{n}{w_{j}x_{j}} \right) + b
$$

```python
f = 0 
for j in range(0, n):
    f = f + w[j] * x[j]
f = f + b
```

#### Using Vectorization

$$
f_{\vec{w},b}(\vec{\mathbf{x}}) = \vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b \newline
$$

This vectorized implementation using NumPy will run much faster because it will utilize parallel computation.

```python
f = np.dot(w, x) + b
```

[JupyterLab Example](./code/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb)

[<<Previous](../week-01/README.md) | [Next>>]()