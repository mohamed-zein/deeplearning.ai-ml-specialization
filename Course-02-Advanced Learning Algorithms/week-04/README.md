# Decision trees
## Decision trees
### Decision tree model
![Decision tree](./images/decision-tree-01.jpg)
* The above is an example diagram of a Decision Tree.
* Every _oval_ or _rectangular_ is called a **node**.
    * The topmost first node of the tree is called **root node**.
    * All other nodes (_oval_) in the tress excluding the last level are called **decision nodes**.
    * The last nodes in the tree (_rectangular_) are called **leaf nodes**.
* The way this model works with each text example is as follows:
    * It start with the **root node** and based on the corresponding feature value in the test example, it will decide to go left or right branch in the tree.
    * Then to go down the tree and check the corresponding feature against the **decision node** and decides to go left ot right branch.
    * When it reaches the **leaf node**, it will infer a classification for the test example.

### Learning Process
* Through the process of building decision tree, there were a couple of key decisions that we had to make at various steps during the algorithm.
    1. How to choose what features to split on at each node?
        * Decision trees will choose what feature to split on in order to try to **maximize purity**. 
        * By **purity**, we mean in our example, you want to get to what subsets, which are as close as possible to all cats or all dogs.
    2. When do you stop splitting?
        * When a node is 100% one class.
        * When splitting a node will result in the tree exceeding a maximum depth.
            * One reason you might want to limit the depth of the decision tree is to make sure for us to tree doesn't get too big and unwieldy 
            * And second, by keeping the tree small, it makes it less prone to overfitting.
        ![Decision Tree Depth](./images/decision-tree-02.jpg)
        * When improvements in purity score are below a threshold.
        * When a number of examples in a node is below a threshold.
## Decision tree learning
### Measuring purity
* Let's assume $p_{1}$ = fraction of examples that are $\mathbf{y} = 1$
* We will introduce the Entropy function $H(p_{1})$ as a measure of impurity.
![Entropy](./images/entropy-01.jpg)
* Formula for the entropy function $H(p_{1})$:

$$
\begin{align*}
p_{1} & = \text{fraction of examples that are } \mathbf{y} = 1 \newline
p_{0} & = 1 - p{1} \newline
H(p_{1}) & = -p_{1} \log_{2}(p_{1}) - p_{0} \log_{2}(p_{0}) \newline
& = -p_{1} \log_{2}(p_{1}) - (1-p_{1})\log{2}(1-p_{1})
\end{align*}
$$

### Choosing a split: Information Gain
* When building a decision tree, the way we'll decide what feature to split on at a node will be based on what choice of feature reduces entropy the most (Reduces entropy, reduces impurity or maximizes purity).
* In decision tree learning, the reduction of entropy is called **Information Gain**.
* When choosing a feature to use to split in a node, we edn up with 2 sub-branches.
* Associated with each of these splits is two numbers, the entropy on the left sub-branch and the entropy on the right sub-branch.
* The way we're going to combine these two numbers is by taking a weighted average.
    * Because how important it is to have low entropy in, say, the left or right sub-branch also depends on how many examples went into the left or right sub-branch.
    * If there are lots of examples in, say, the left sub-branch then it seems more important to make sure that that left sub-branch's entropy value is low.
* To follow the convection in decision trees, rather than computing this weighted average entropy, we're going to compute the reduction in entropy compared to if we hadn't split at all which is **Information Gain**.

$$
\text{Information Gain} = H({p_{1}}^{\text{root}}) - \left(w^{\text{left}} H({p_{1}}^{\text{left}}) + w^{\text{right}} H({p_{1}}^{\text{right}}) \right)
$$

> **Why use Information Gain instead of Entropy?**  
> one of the stopping criteria for deciding when to not bother to split any further is if the reduction in entropy is too small. In which case you could decide, you're just increasing the size of the tree unnecessarily and risking overfitting by splitting and just decide to not bother if the reduction in entropy is too small or below a threshold.

### Putting it together
#### Decision Tree Learning
* Start with all examples at the root node.
* Calculate the information gain for all possible features, and pick one with the highest information gain.
* Split dataset according to selected feature, and create left and right branches of the tree.
* Keep repeating splitting process until stopping criteria is met:
    * When node is 100% one class.
    * When splitting a node will result in the tree exceeding a maximum depth.
    * Information gain from additional splits is less than threshold.
    * When a number of examples in a node is below a threshold.
* Implementing a decision tree utilize Recursion.
> **Recursion**  
> Recursion in computer science refers to writing code that calls itself.

### Using one-hot encoding of categorical features
* **One-Hot Encoding** is used with features that can take on more than two values.
> **One-Hot Encoding**  
> If a categorical feature can take on $k$ possible values then we will replace it by creating $k$ binary features that can only take on the values $0$ or $1$.
* **One-Hot Encoding** is also useful applying neural networks on categorical features.
### Continuous valued features
* To get the decision tree to work on continuous value features at every node. When consuming splits:
    * You would just consider different values to split on.
    * Carry out the usual information gain calculation.
    * Decide to split on that continuous value feature if it gives the highest possible information gain.

### Regression Trees
* We'll generalize decision trees to be regression algorithms so that we can predict a number.
* When building a regression tree, rather than trying to reduce [entropy](#measuring-purity), which was that measure of impurity that we had for a classification problem, we instead try to reduce the _variance of the weight_ of the values $Y$ at each of these subsets of the data.
#### Choosing a split
* A good way to choose a split would be to just choose the value of the weighted variance that is lowest.
* For a regression tree we'll measure the reduction in variance.

[Lab: Decision Trees](./code/C2_W4_Lab_01_Decision_Trees.ipynb)

[<<Previous](../week-03/README.md) | [Next>>]()