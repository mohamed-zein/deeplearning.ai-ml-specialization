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

[<<Previous](../week-03/README.md) | [Next>>]()