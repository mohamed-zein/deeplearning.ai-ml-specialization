# Advice for applying machine learning
## Advice for applying machine learning
#### Debugging a learning algorithm
* Recall the regularized linear regression cost function:

$$
J(\vec{\mathbf{w}}, b) = \frac{1}{2m}\sum\limits_{i=1}^{m}{\left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}^{(i)}\right) - \mathbf{y}^{(i)}\right)^{2}} + \frac{\lambda}{2m} \sum\limits_{j=1}^{n}{\mathbf{w}^{2}_{j}}
$$

* If the model makes unacceptably large errors in prediction, we can try:
    * Get more training example.
    * Try smaller set of features.
    * Try getting additional features.
    * Try adding polynomial features $(x_{1}^{2}, x_{2}^{2}, x_{1}, x_{2}, \text{etc})$.
    * Try decreasing $\lambda$
    * Try increasing $\lambda$
#### Machine Learning Diagnostics
* **Diagnostic**: A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance.
### Evaluating a model
* **Overfitting**: Model fits the training data well but will fail to generalize to new examples not in the training set.
* We need some more systematic way to evaluate how well your model is doing. One technique, we can use is to split the training set into two subsets.:
    * 70% of the data to be **training set** $m_{train} = \text{No. training examples}$.
    * 30% of the data to be **testing set** $m_{test} = \text{No. test examples}$.
#### Train/test procedure for linear regression (with squared error cost)
* Fit parameters by minimizing cost function $J(\vec{\mathbf{w}},b)$ to find $\vec{\mathbf{w}},b$:

$$
J(\vec{\mathbf{w}}, b) = \frac{1}{2m_{train}}\sum\limits_{i=1}^{m_{train}}{\left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}^{(i)}\right) - \mathbf{y}^{(i)}\right)^{2}} + \frac{\lambda}{2m_{train}} \sum\limits_{j=1}^{n}{\mathbf{w}^{2}_{j}}
$$

* Compute test error:

$$
J_{test}(\vec{\mathbf{w}}, b) = \frac{1}{2m_{test}}\sum\limits_{i=1}^{m_{test}}{\left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{test}^{(i)}\right) - \mathbf{y}_{test}^{(i)}\right)^{2}}
$$

* Compute training error:

$$
J_{train}(\vec{\mathbf{w}}, b) = \frac{1}{2m_{train}}\sum\limits_{i=1}^{m_{train}}{\left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{train}^{(i)}\right) - \mathbf{y}_{train}^{(i)}\right)^{2}}
$$

#### Train/test procedure for classification problem
* Fit parameters by minimizing cost function $J(\vec{\mathbf{w}},b)$ to find $\vec{\mathbf{w}},b$:

$$
J(\vec{\mathbf{w}}, b) = -\frac{1}{m_{train}}\sum\limits_{i=1}^{m_{train}}{\left[\mathbf{y}^{(i)} \log \left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}^{(i)}\right) \right) + \left(1- \mathbf{y}^{(i)} \right) \log \left(1-f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}^{(i)}\right)\right) \right]} + \frac{\lambda}{2m_{train}}\sum\limits_{j=1}^{n}{\mathbf{w}_{j}^{2}}
$$

* Compute test error:

$$
J_{test}(\vec{\mathbf{w}}, b) = -\frac{1}{m_{test}}\sum\limits_{i=1}^{m_{test}}{\left[\mathbf{y}_{test}^{(i)} \log \left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{test}^{(i)}\right) \right) + \left(1- \mathbf{y}_{test}^{(i)} \right) \log \left(1-f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{test}^{(i)}\right)\right) \right]}
$$

* Compute training error:

$$
J_{train}(\vec{\mathbf{w}}, b) = -\frac{1}{m_{train}}\sum\limits_{i=1}^{m_{train}}{\left[\mathbf{y}_{train}^{(i)} \log \left(f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{train}^{(i)}\right) \right) + \left(1- \mathbf{y}_{train}^{(i)} \right) \log \left(1-f_{\vec{\mathbf{w}}, b}\left(\vec{\mathbf{x}}_{train}^{(i)}\right)\right) \right]}
$$

### Model selection and training/cross validation/test sets
* When having multiple models to select from, using only $J_{test}(\vec{\mathbf{w}}, b)$ is likely to be an optimistic estimate of generalization error because it relies only on the test set.
* To select a model out of possible multiple models, we need to modify the procedure by splitting our data into 3 different datasets:
    * **Training Set**: $m_{train} \approx 60\%$ of the data.
    * **Cross Validation Set**: $m_{cv} \approx 20\%$ of the data. Other names of this data set:
        * _Validation Set_.
        * _Development Set_.
        * _Dev Set_.
    * **Test Set**: $m_{train} \approx 20\%$ of the data.
* So to select a model, this can be done based on the Cross Validation error $J_{cv}(\vec{\mathbf{w}}, b)$.
* To estimate generalization error, this can be done using the Test error $J_{test}(\vec{\mathbf{w}}, b)$


[Lab: Model Evaluation and Selection](./code/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb)

[<<Previous](../week-02/README.md) | [Next>>]()