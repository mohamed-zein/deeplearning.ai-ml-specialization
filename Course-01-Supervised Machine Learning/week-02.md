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

[<<Previous](./week-01.md) | [Next>>]()