# EM + GMM
## Programming1
Consider the case that the hidden variable $y \in \{1, ..., m\}$ is discrete while the visible variable $x \in R^d$ is continuous. 
In other words, we consider mixture models of the form
$$
p(x) = \sum_{j = 1}^m p(x|y = j)p(y=j).
$$
We assume throughout that $x$ is conditionally Gaussian in the sense that $x \sim \mathcal{N}(\mu_j, \Sigma_j)$ when $y = j$.

The file **emdata.mat** contains visible variable $x$.
Please implement EM algorithm manually, and:
- Run EM algorithm on this data with $m=2,3,4,5$ and visualize results.
- Modify the M-step such that the covariance matrices of different Gaussian components are equal. Give derivation and rerun the code with $m=2,3,4,5$ and provide visualizations.

In this assignment, you are **NOT** allowed to use any existing libraries or code snippets that provides EM algorithm (e.g., **scikit**,  **OpenCV** in python or **emgm** in Matlab).

##Programming 2 (Missing Data)
| Point |   $x_1$   |   $x_2$   |  $x_3$   |
| :---: | :----: | :----: | :---: |
|   1   |  0.42  | -0.087 | 0.58  |
|   2   |  -0.2  |  -3.3  | -3.4  |
|   3   |  1.3   | -0.32  |  1.7  |
|   4   |  0.39  |  0.71  | 0.23  |
|   5   |  -1.6  |  -5.3  | -0.15 |
|   6   | -0.029 |  0.89  | -4.7  |
|   7   | -0.23  |  1.9   |  2.2  |
|   8   |  0.27  |  -0.3  | -0.87 |
|   9   |  -1.9  |  0.76  | -2.1  |
|  10   |  0.87  |  -1.0  | -2.6  |


Suppose we know that the ten data points in category $\omega_1$ in the table above come from a three-dimensional Gaussian. Suppose, however, that we do not have access to the $x_3$ components for the even-numbered data points.

1. Write an EM program to estimate the mean and covariance of the distribution. Start your estimate with $\pmb{\mu}^0=0$ and $\pmb{\Sigma}^0 = \textbf{I} $, the three-dimensional identity matrix.

2. Compare your final estimation with that for the case when there are no missing data.