# Ensemble

## Programming 1: Adaboost Programming
The goal of this problem is to give you an overview of the procedure of **Adaboost.**

Here, our "weak learners" are **decision stumps**. Our data consist of $X \in \mathbb{R}^{n\times p}$ matrix with each row a sample and label vector $y\in\{-1, +1\}^{n}$. A decision stump is defined by:
$$
h_{(a,d,j)}(\bold{x})= \left\{\begin{array}{ll}
d & if\ x_{j}\leq a \\
-d & otherwise
\end{array} \right.
$$
Where $a\in \mathbb{R}$, $j\in \{1, ..., p\}$, $d\in \{-1, +1\}$. Here $\bold{x}\in \mathbb{R}^{p}$ is a vector, and $x_{j}$ is the $j$-th coordinate.

Directory of the data is \emph{$/code/ada\_data.mat$}. It contains both a training and testing set of data. Each consists of 1000 examples. There are 25 real valued features for each example, and a corresponding $y$ label.

2.1 Complete the code skeleton $\bold{decision\_stump.m}$ (or $\bold{decision\_stump()}$ in adaboost.py if you use python). This program takes as input: the data along with a set of weights (i.e., $\{(\bold{x}_{i}, y_{i}, w_{i})\}_{i=1}^{n}$, where $w_{i} \geq 0$ and $\sum_{i=1}^{n}w_{i}=1$), and returns the decision stump which minimizes the weighted training error. Note that this requires selecting both the optimal $a$, $d$ of the stump, and also the optimal coordinate $j$.

The output should be a pair $(a^{\star}, d^{\star}, j^{\star})$ with:
$$
l(a^{\star}, d^{\star}, j^{\star})=min_{a, d, j}l(a, d, j)=min_{a, d, j}\sum_{i=1}^{n}w_{i}1\{h_{a, d, j}(\bold{x}_{i})\neq y_{i}\}
$$
Your approach should run in time $O(pn\ log\ n)$ or better. Include details of your algorithm in the report and analyze its running time.

\emph{Hint: you may need to use the function $\bold{sort}$ provided by matlab or python in your code, we can assume its running time to be $O(m log m)$ when considering a list of length m.}

2.2 Complete the other two code skeletons $\bold{update\_weights.m}$ and $\bold{adaboost\_error.m}$. Then run the $\bold{adaboost.m}$, you will carry out adaboost using decision stumps as the "weak learners". (Complete the code in $\bold{adaboost.py}$ if you use python)

2.3 Run your AdaBoost loop for 300 iterations on the data set, then plot the training error and testing error with iteration number as the x-axis.

## Programming 2: Random Forest Programming

A random forest is a bagging ensemble of a number of decision trees on various sub-samples of the dataset to improve the predictive accuracy and control over-fitting. In this problem, you will implement random forest algorithm based on your decision tree algorithm for Problem 3 in Homework 11.

The basic requirements are the same as Homework 11:
- Do not use any existing libraries.
- Use the same splits for training, validation, and testing as Homework 11. Report the hyper-parameters and the performance on the test set of your best model which is selected by performance on the validation set.

3.1 A random forest consists of $n_{tree}$ decision trees. Each tree in the ensemble is built from a sub-sample drawn with replacement (i.e., a bootstrap sample) from the training set. The sub-sample size is the same as the original input sample size of the training set. The prediction of the forest is given as the majority vote of the individual classifiers. 

Implement random forest algorithm and compare its performance with a single decision tree trained without boostrap. Note the best hyper-parameters for individual trees in the random forest might differ from those for a single decision tree.

3.2 To increase randomness, when splitting each node during the construction of a tree, select a random subset of size $max\_features$ from all input features (without replacement) and choose the best feature from the subset.

Implement random forest algorithm with randomness on features and compare it with the forest trained without randomness on features.

3.3 The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking a majority vote of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.

Repeat training random forests with different random seeds and compare the mean and variance of testing accuracy of random forests and the individual trees.