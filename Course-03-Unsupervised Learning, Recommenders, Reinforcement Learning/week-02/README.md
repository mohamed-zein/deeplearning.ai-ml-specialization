# Recommender systems
## Collaborative filtering
### Making recommendations
* In the typical recommended system, we have some number of **users** as well as some **number of items**.
* When users rate items they already used, then it is possible to predict the ratings of other items they didn't rate and hence recommend those items to them.
* The notation we use:
    * $n_{u} =$ no. of users.
    * $n_{m} =$ no. of movies.
    * $r(i, j) = 1$ if user $j$ has rated movie $i$.
    * $\mathbf{y}^{(i, j)} =$ rating given by user $j$ to movie $i$ (defined only if $r(i, j) = 1$).
### Using per-item features
* To develop our recommendations system, we will add features to the items.
    * In case of movies, we can add features related to the genre of each movie.
    * We will donate $n$ as the number of features that each items has.
* If the item has $n=2$ features then we can donate the features of the first item as $\mathbf{x}^{(1)} = \begin{bmatrix} 0.9 \newline 0 \end{bmatrix}$
* So for user 1, to predict the rating for movie $i$, we use a formula similar to _linear regression_ as: $\mathbf{w}^{(1)} \cdot \mathbf{x}^{(i)} + b^{(1)}$
* And to generalize, for user $j$ the formula would be: $\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}$.
* The difference with _Linear Regression_ is in Recommendation system, we fit a Linear Regression model for each user in the dataset.
#### Cost Function
* Notation:
    * $r(i, j) = 1$ if user $j$ has rated movie $i$ ($0$ otherwise).
    * $\mathbf{y}^{(i, j)} =$ rating given by user $j$ to movie $i$ (if defined).
    * $\mathbf{w}^{(j)}, b^{(j)} =$ parameters for user $j$.
    * $\mathbf{x}^{(i)} =$ feature vector for movie $i$.
    * For user $j$ and movie $i$, predict rating: $\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}$.
    * $m^{(j)} =$ no. of movies rated by user $j$.
* To learn $\mathbf{w}^{(j)}, b^{(j)}$, the cost function to minimize will be:

$$
\min_{\mathbf{w}^{(j)}, b^{(j)}} J(\mathbf{w}^{(j)}, b^{(j)}) = \frac{1}{2} \sum\limits_{i:r(i,j)=1}{ \left( \underbrace{\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(i)}}_{\text{Predicted rating}} - \underbrace{\mathbf{y}^{(i,j)}}_{\text{Actual rating}} \right)^{2} } + \underbrace{\frac{\lambda}{2} \sum\limits_{k=1}^{n}{ \left( \mathbf{w}_{k}^{(j)} \right)^{2} }}_{\text{Regularization term}}
$$

* To learn parameters $\mathbf{w}^{(1)}, b^{(1)}, \dots , \mathbf{w}^{(n_{u})}, b^{(n_{u})}$ for all users:

$$
\min_{\mathbf{w}^{(1)}, b^{(1)}, \dots , \mathbf{w}^{(n_{u})}, b^{(n_{u})}} J \begin{pmatrix} \mathbf{w}^{(1)}, & \dots, & \mathbf{w}^{(n_{u})} \newline b^{(1)}, & \dots, & b^{(n_{u})} \end{pmatrix} = \frac{1}{2} \sum\limits_{j=1}^{n_{u}} \sum\limits_{i:r(i,j)=1}{ \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(i)} - \mathbf{y}^{(i,j)} \right)^{2} } + \frac{\lambda}{2} \sum\limits_{j=1}^{n_{u}} \sum\limits_{k=1}^{n}{ \left( \mathbf{w}_{k}^{(j)} \right)^{2} }
$$
### Collaborative filtering algorithm
* What if we don't have features, $x_1$ and $x_2$ of the product items?
![Problem Motivation](./images/recomendation-01.jpg)
* In collaborative filtering, because we have ratings from multiple users of the same item with the same movie, it is possible guess features $x_1$ and $x_2$ from scratch.
#### Cost Function for estimating $\mathbf{x}^{(i)}$
* Given $\mathbf{w}^{(1)}, b^{(1)}, \mathbf{w}^{(2)}, b^{(2)}, \dots , \mathbf{w}^{(n_{u})}, b^{(n_{u})}$, to learn $\mathbf{x}^{(i)}$:

$$
J\left(\mathbf{x}^{(i)}\right) = \frac{1}{2} \sum\limits_{j:r(i,j) = 1}{ \left(\underbrace{\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}}_{\text{Predicted}} - \underbrace{\mathbf{y}^{(i,j)}}_{\text{Actual}} \right)^{2} } + \underbrace{\frac{\lambda}{2} \sum\limits_{k=1}^{n}{ \left( \mathbf{x}_{k}^{(i)} \right)^{2}} }_{\text{Regularization Term}}
$$

* So therefore all the users $j$ that have rated movie $i$, we will try to minimize the squared difference between what your choice of features $\mathbf{x}^{(i)}$ results in terms of the predicted movie rating minus the actual movie rating that the user had given it.
    * Finally, if we want to add a regularization term, we add the usual $+ \frac{\lambda}{2}$, $k= 1$ through $n$, where $n$ as usual is the number of features of $\mathbf{x}^{(i)}$ squared.

* To learn all the features in our data set $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}. \dots , \mathbf{x}^{(n_{m})}$:

$$
\min_{\mathbf{x}^{(1)},\dots , \mathbf{x}^{(n_m)}} J\left(\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots , \mathbf{x}^{(n_m)}\right) = \frac{1}{2} \sum\limits_{i=1}^{n_{m}} \sum\limits_{j:r(i,j) = 1}{ \left(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - \mathbf{y}^{(i,j)} \right)^{2} } + \frac{\lambda}{2} \sum\limits_{i=1}^{n_{m}} \sum\limits_{k=1}^{n}{ \left( \mathbf{x}_{k}^{(i)} \right)^{2}}
$$


#### Final Cost function
* We assumed we had those parameters $\mathbf{w}$ and $\mathbf{b}$ for the different users. Where do you get those parameters from? We can put together the [cost function for parameters](#cost-function) with the [cost function for features](#cost-function-for-estimating).

$$
\min_{\mathbf{w}^{(1)}, \dots , \mathbf{w}^{(n_{u})}, b^{(1)}, \dots , b^{n_{u}} \mathbf{x}^{(1)},\dots , \mathbf{x}^{(n_m)}} J(\mathbf{w}, b,  \mathbf{x}) = \frac{1}{2} \sum\limits_{(i,j):r(i,j) = 1}{ \left(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - \mathbf{y}^{(i,j)} \right)^{2} } + \frac{\lambda}{2} \sum\limits_{i=1}^{n_{m}} \sum\limits_{k=1}^{n}{ \left( \mathbf{x}_{k}^{(i)} \right)^{2}} + \frac{\lambda}{2} \sum\limits_{j=1}^{n_{u}} \sum\limits_{k=1}^{n}{ \left( \mathbf{w}_{k}^{(j)} \right)^{2} }
$$

![Gradient Descent](./images/recomendation-02.jpg)
### Binary labels: favs, likes and clicks
* Many applications of recommendation systems or collective filtering algorithms involvs binary labels.
* Binary labels is instead of a user giving you a one to five star or zero to five star rating, they just somehow give you a sense of they like this item or they did not like this item.
* The process we'll use to generalize the algorithm will be very much reminiscent to how we have gone from linear regression to logistic regression.
#### Example Applications
1. Did user $j$ purchase an item after being shown?
2. Did user $j$ fav/like an item?
3. Did user $j$ spend at least 30sec with an item?
4. Did user $j$ click on an item?

Meaning of rating:
* 1 - engaged after being shown an item.
* 0 - did not engage after being shown item.
* ? - item not yet shown.

#### Cost function for binary classification
* Loss for binary labels $\mathbf{y}^{(i,j)} = f_{(\mathbf{w}, b, \mathbf{x})} (\mathbf{x}) = g(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)})$

$$
L \left(f_{(\mathbf{w}, b, \mathbf{x})} (\mathbf{x}), \mathbf{y}^{(i,j)} \right) = -\mathbf{y}^{(i,j)} \log{\left( f_{(\mathbf{w}, b, \mathbf{x})} (\mathbf{x}) \right)} - \left(1 - \mathbf{y}^{(i,j)} \right)\log{\left(1- f_{(\mathbf{w}, b, \mathbf{x})} (\mathbf{x}) \right)}
$$

* So the cost function would be:

$$
J(\mathbf{w}, b, \mathbf{x}) = \sum\limits_{(i,j):r(i,j) = 1}{L \left(\underbrace{f_{(\mathbf{w}, b, \mathbf{x})} (\mathbf{x})}_{g(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)})}, \mathbf{y}^{(i,j)} \right)}
$$

## Recommender systems implementation detail


[<<Previous](../week-01/README.md) | [Next>>]()