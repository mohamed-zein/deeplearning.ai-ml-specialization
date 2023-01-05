# Supervised Machine Learning: Regression and Classification
## What is machine learning?
Field of study that gives computers the ability to learn without being explicitly programmed.
### Machine Learning algorithms
* Supervised learning: 
    * Used most in real-world applications.
    * Has seen rapid advancements and innovations.
    * In this course, we will focus on this type.
* Unsupervised learning
* Recommender systems
* Reinforcement learning
## Supervised learning
* Refers to algorithms that learn `x` to `y` or `input` to `output` mappings.  
* The key characteristics of supervised learning is that you give your learning algorithm examples to learn from that includes the `right answers`.  
* By seeing correct pairs of `input x` and desired `label y`, the learning algorithm eventually learns to take the input alone without the output label and gives a reasonably accurate prediction or guess of the output.
### Examples
Input (X) | Output(Y) | Application
----------|-----------|------------
email | spam?(0/1) | spam filtering
audio | text transcripts | speech recognition
English | Spanish machine | translation
ad, user info | click?(0/1) | online advertising
image, radar info | position of other cars | self-driving car
image of phone | defect?(0/1) | visual inspection

### Regression
**Regression** is a special type of Supervised Learning which predicts a **number** from infinitely many possible outputs.
#### Regression: Housing price prediction
![Housing price prediction](./images/regression-01.jpg)
* Here we gather data about houses and plot their size against their prices.
* Then we try to fit a line or a curve (based on a more complicated function) to the data point.
* Then we can predict the price of a house with size not in our training dataset.

### Classification
* In **Classification**, the algorithm tries use the input to predict the output _class_ or _category_ out of small number of possible classes/categories.
    * This is different from [Regression](#regression) which tries to predict any number of infinitely many possible numbers.
![Classification: breast cancer detection](./images/classification-01.jpg)
* In Classification, the terms output **Class** or **Category** are often used interchangeably.
* Classification algorithms predict categories.
* Categories don't have to be numbers.

#### Example
When ploting the data of breast cancer (_Age_ & _Tumor size_ vs _Tumor Type_)
![Classification: breast cancer detection - 2 Inputs](./images/classification-02.jpg)
* We use circles to represent patients with benign tumor and corsses to represent patients with malignant tumor.
* So when a new patient comes in, the doctor measures the tumor size and record the age.
* Given a dataset like this, the learning algorithm tries to find some boundry that seperates the malignant tumors from the benign ones. So the learning algorithm has to decide how to fit a boundry line between that classify the data.
