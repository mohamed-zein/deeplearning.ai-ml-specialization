# Unsupervised learning
## Clustering
### What is clustering?
* A clustering algorithm looks at a number of data points and automatically finds data points that are related or similar to each other.
* Clustering is an **unsupervised learning** algorithm
    * In _supervised learning_, the dataset included both the inputs $\mathbf{x}$ as well as the target outputs $\mathbf{y}$.
    * In _unsupervised learning_, you are given a dataset like this with just $\mathbf{x}$, but not the labels or the target labels $\mathbf{y}$.
    * Because we don't have target labels $\mathbf{y}$, we're not able to tell the algorithm what is the **right answer**, $\mathbf{y}$ that we wanted to predict.
    * Instead, we're going to ask the algorithm to find something interesting about the data, that is to find some interesting structure about this data. 
#### Applications of clustering
* Grouping simalar news.
* Market segmentation.
* DNA Analysis.
* Astronomical data analysis.
### K-means intuition
* K-means is a clustering algorithm.
* Initially K-means starts by taking a random guess at where might be the centers of the two clusters that you might ask it to find.
    > **Centriods**: the centers of the cluster are called cluster _Centroids_.
* After it has made an initial guess at where at where the cluster centroid is, it will go through all of data points.
    * And for each of them it will check which centroid is closer to this data point.
    * And it will assign each of these points to whichever of the cluster centroids It is closer to.
    * It will move each centroid to whatever is the average location of the data points associated with it.
* K-means will repeatedly do two different things:
    1. The first is assign points to cluster centroids.
    2. The second is move cluster centroids.
### K-means algorithm
* Randomly initialize $K$ cluster centroids $\mu_{1}, \mu_{2}, \dots, \mu_{K}$.
* Repeat:
    * Assign points to cluster centroids
    * Move cluster centroids

    > **Corner Case** if a cluster has zero training examples assigned to it. In that case, the second step, $\mu_{K}$, would be trying to compute the average of zero points which is not defined.  
    > The most common thing to do is to just eliminate that cluster. You end up with $K - 1$ clusters.  
    > Alternatively, we can randomly reinitialize that cluster centroid and hope that it gets assigned at least some points next time round.
### Optimization objective
> **Notations**  
> * $c^{(i)} = \text{index of cluster } (1, 2, \dots , K) \text{ to which example } x^{(i)} \text{ is currently assigned}$
> * $\mu_{k} = \text{Cluster centroid } k$
> * $\mu_{c^{(i)}} = \text{Cluster of cluster to which example } x^{(i)} \text{has been assigned}$

#### Cost Function
$$
J(c^{(1)}, \dots , c^{(m)}, \mu_{1}, \dots, \mu_{K}) = \frac{1}{m} \sum\limits_{i=1}^{m}{\left\| x^{(i)} - \mu_{c^{(i)}} \right\|}^{2}
$$

![Cost Function for K-means](./images/kmeans-01.jpg)

> In some context, the above formula is called **Distortion Function**.

### Initializing K-means
#### Random Initialization
* Random Initialization steps:
    1. First choose $K < m$.
    2. Randomly pick $K$ training examples.
    3. Set $\mu_{1}, \mu_{2}, \dots , \mu_{K}$ equal to this $K$ examples.
* Depending on how you choose the random initial centroids, K-means will end up picking a difference set of centroids for your data set.
    * With less fortunate choice of random initialization, it can happen to get stuck in a local minimum.
![Random Initialization](./images/kmeans-02.jpg)
* To give K-means multiple shots at finding the best local optimum, One other thing you could do with the K-means algorithm is to run it multiple times and then to try to find the best local optima.
* If you were to run K-means 3 times say, then one way to choose between these 3 solutions, is to compute the cost function $J$ for all 3 of these solutions. And then to pick the one which gives you the lowest value for the cost function $J$.
![Random Initialization Algorithm](./images/kmeans-03.jpg)

### Choosing the number of clusters
* The K-means algorithm requires as one of its inputs, $K$, the number of clusters you want it to find.
#### What is the right value of $K$?
* For a lot of clustering problems, the right value of $K$ is truly ambiguous.
* Because clustering is unsupervised learning algorithm you're not given the _right answers_ in the form of specific labels to try to replicate.
* There are lots of applications where the data itself does not give a clear indicator for how many clusters there are in it.
#### Choosing the value of $K$
* The right value of $K$ is truly ambiguous but in some applications, it is possible to use the [Elbow method](#elbow-method)
* [Elbow method](#elbow-method) is not always optimal so it is recomended to evaluate K-means based on how well it performs for later purposes.
##### Elbow Method
![Elbow method](./images/kmeans-04.jpg)
* In this method:
    1. You would run K-means with a variety of values of $K$
    2. Plot the cost function $J$ as a function of the number of clusters.

[Assignment Lab: k-means](./assignment-kmeans/C3_W1_KMeans_Assignment.ipynb)
----------


[<<Previous](../README.md) | [Next>>]()