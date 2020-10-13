# Error + Linear
## Problem 4: Programming for Error Rate Estimation
Consider a two dimensional classification problems: $p(\omega_{1})=p(\omega_{2})= 0.5$, $p(x|\omega_{1}) \sim N(\mu_{1},\sigma_{1}),p(x|\omega_{2}) \sim N(\mu_{2},\sigma_{2})$, where $\mu_{1} = (-1,0),\mu_{2} = (1,0),\sigma_{1} = (\begin{matrix}
1&0\\
0&1
\end{matrix}),\sigma_{2} = (\begin{matrix}
2&0\\
0&1
\end{matrix})$

4.1 Find the misclassification rate of Bayesian classifier numerically.

4.2 Draw $n$ samples from $p(x|\omega_{1})$ and $p(x|\omega_{2})$ with labels respectively. 
Estimate $p_{n}(x|\omega_{1})$ and $p_{n}(x|\omega_{2})$ by Parzen window method, with Gaussian window function and unit hypercube window function. 
Design Bayesian classifier with your estimated $p_{n}(x|\omega_{1})$ and $p_{n}(x|\omega_{2})$. 
Compare their misclassification rate with the theoretical optimal Bayesian classifier in theory.

4.3 From above experiments, what's your suggestion for choosing optimal window function and parameters with given $n$?

4.4 Sample $2n$ points from the mixture Gaussian distribution p(x) without labels. Use EM to estimate$\mu_{1},\mu_{2},\sigma_{1},\sigma_{2}$ so that we estimate $p_{2n}(x|\omega_{1})$ and $p_{2n}(x|\omega_{2})$. Which method is more accurate in estimating $p(x|\omega_{1})$ and $p(x|\omega_{2})$, EM or Parzen window? Prove your statement by experiments.

4.5 Design Bayesian classifier with the estimated $p_{2n}(x|\omega_{1})$ and $p_{2n}(x|\omega_{2})$ by EM. Analyze its performance, i.e., the expectation and variance of misclassification rate and compare them with that of optimal Bayesian classifier.

4.6 Conclude your results. Which method is your favorite to estimate parameters and which classifier is your favorite classifier? Why?

## Problem 5: Programming for Perceptron Algorithm
The training process of the classical perceptron (Algorithm~\ref{alg:1}) can be regarded as a searching process for a solution in feasible solution region, whereas no strict restrictions
are demanded for the capacity of this solution. The solution only needs to satisfy $\alpha^Ty_n>0$, where $\alpha$ is the weight vector of the perceptron, and $y_n$ is the normalized augmented sample vector. However, the margin perceptron (Algorithm~\ref{alg:2}) requires the finally converged hyperplane possesses a margin ($>\gamma$), where $\gamma$ is a predifined positive scalar. It means that the final solution of perceptron need to satisfy   $\alpha^Ty_n>\gamma$.

Thus, there are two types of ``mistakes'' during the traing of perceptron, namely (1) the prediction mistake and (2) margin mistake (i.e., its prediction is correct but its margin is not large enough).


\begin{algorithm}[htb]
  \caption{Fixed-Increment Single Sample Correction Algorithm}
  \label{alg:1}
  \begin{algorithmic}[1]
  \STATE \textbf{initialize} $\alpha$,$k \gets 0$
  \REPEAT
    \STATE $k \gets (k+1)$ $mod$ $n$
    \IF{$y_k$ is misclassified by $\alpha$}
        \STATE $\alpha \gets \alpha + y_k$
    \ENDIF
  \UNTIL {all patterns are properly classified}
  \STATE \textbf{return} $\alpha$
  \end{algorithmic}
\end{algorithm}

\begin{algorithm}[htb]
  \caption{Single Sample Correction Algorithm With Margin}
  \label{alg:2}
  \begin{algorithmic}[1]
  \STATE \textbf{initialize} $\alpha,k \gets 0,~\gamma > 0$
  \REPEAT
    \STATE $k \gets (k+1)$ $mod$ $n$
    \IF{$\alpha^Ty_k\leq\gamma$}
        \STATE $\alpha \gets \alpha + y_k$
    \ENDIF
  \UNTIL {all patterns are properly classified with a large enough margin $\gamma$}
  \STATE \textbf{return} $\alpha$
  \end{algorithmic}
\end{algorithm}

5.1 Please generate 200 datapoints in the 2D plane, among which 100 datapoints are labeled as 1 and the remaining 100 are labeled as -1. Make sure that these 200 datapoints are linearly separable. Plot these 200 datapoints in a 2D plane;

5.2 Implement the classical perceptron algorithm and run it on the above generated datapoints. Plot the classification boundary and these datapoints in one figure;

5.3 Implement the margin perceptron algorithm and run it on the above generated datapoints. Plot the classification boundary and these datapoints in one figure. Analyse the impacts of $\gamma$ on algorithm convergence and the classification boundary.
