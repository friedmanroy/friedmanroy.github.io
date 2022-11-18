---
layout: distill
title: 5 - Equivalent Form for Bayesian Linear Regression
description: Follows the construction of an equivalent form for Bayesian LR

authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University

toc:
  - name: Woodbury Matrix Inversion
  - name: Equivalent Form
  - name: Regular vs Equivalent
---

In the [previous post](https://friedmanroy.github.io/BML/rec_4/), we saw that the posterior distribution of Bayesian linear regression is given by:

$$
\begin{equation}
p\left(\theta\,\mid \,y\right)=\mathcal{N}\left(y\,\mid \,\mu_{\theta\mid D},C_{\theta\mid D}\right)
\end{equation}
$$
where:
$$
\begin{align}
\mu_{\theta\mid D} & =C_{\theta\mid D}\left(H^{T}\frac{1}{\sigma^{2}}y+\Sigma_{\theta}^{-1}\mu_{\theta}\right)\\
C_{\theta\mid D} & =\left(\Sigma_{\theta}^{-1}+\frac{1}{\sigma^{2}}H^{T}H\right)^{-1}
\end{align}\label{eq:post-cov}
$$

However, the posterior has an equivalent form which is sometimes used in literature. To find this equivalent form, we will need to talk about the _Woodbury matrix identity_.


## Woodbury Matrix Identity

The identity is given by:

$$
\begin{equation}
\left(A+UCV\right)^{-1}=A^{-1}-A^{-1}U\left(C^{-1}+VA^{-1}U\right)^{-1}VA^{-1}\label{eq:woodbury}
\end{equation}
$$
where we assumed that $A$ and $C$ are invertible, while $U$ and $V$ don't even have to be square. Before we continue to use this on the covariance of the posterior, we should talk about when to use this identity. Obviously, if all of the matrices $A$ and $C$ are square and have the same dimensions and we know nothing about their inverses, using this identity will not help us at all. However, many times we will be confronted with an equation similar to the left hand side of equation \eqref{eq:woodbury}, where we actually know what the inverse of $A$ is directly, or know that the inverses of $A$ and $C$ are rather simple to compute.

### Example: Low Rank Matrices

Let's look at an example. Suppose we want to find:
$$
\begin{equation}
\left(I_{n}\beta+\frac{1}{\alpha}AA^{T}\right)^{-1}
\end{equation}
$$
where $A\in\mathbb{R}^{n\times m}$ such that $n\gg m$; in this sense $AA^{T}$ is a _low rank_ matrix since its rank (at most $m$ ) is much smaller than the full rank ( $n$ ). In this case, inverting the bigger $n\times n$ matrix will be much less efficient than inverting a small $m\times m$ matrix. We can now put the identity to good use:
$$
\begin{align}
\left(I_{n}\beta+\frac{1}{\alpha}AA^{T}\right)^{-1} & =\left(I_{n}\beta\right)^{-1}-\left(I_{n}\beta\right)^{-1}A\left(\left(I_{m}\frac{1}{\alpha}\right)^{-1}+A^{T}\left(I_{n}\beta\right)^{-1}A\right)^{-1}A^{T}\left(I_{n}\beta\right)^{-1}\nonumber \\
 & =\frac{1}{\beta}I_{n}-\frac{1}{\beta^{2}}A\left(I_{m}\alpha+\frac{1}{\beta}A^{T}A\right)^{-1}A^{T}\nonumber \\
 & =\frac{1}{\beta}I_{n}-\frac{1}{\beta}A\left(I_{m}\alpha\beta+A^{T}A\right)^{-1}A^{T}
\end{align}
$$
Notice that the matrix $A^{T}A$ is an $m\times m$ matrix, so we end up only needing to invert $m\times m$ matrices, possibly avoiding many unneeded computations. 

---

# Equivalent Form

We now turn back to the covariance we found in equation \eqref{eq:post-cov}. Using the Woodbury identity:
$$
\begin{align}
C_{\theta\mid D} & =\left(\Sigma_{\theta}^{-1}+\frac{1}{\sigma^{2}}H^{T}H\right)^{-1}\nonumber \\
 & =\Sigma_{\theta}-\Sigma_{\theta}H^{T}\left(\sigma^{2}I+H\Sigma_{\theta}H^{T}\right)^{-1}H\Sigma_{\theta}\\
 & \stackrel{\Delta}{=}\Sigma_{\theta}-\Sigma_{\theta}H^{T}M^{-1}H\Sigma_{\theta}
\end{align}
$$

while this doesn't look particularly helpful, sometimes the number of samples (the first dimension of $H$ ) will be much smaller than the feature space (the dimension of $\Sigma_{\theta}$ ), in which case we will want to invert in the sample dimension. 

It will also be helpful to look at the mean of the posterior in this notation:
$$
\begin{align}
\mu_{\theta\mid D} & =C_{\theta\mid D}\left(H^{T}\frac{1}{\sigma^{2}}y+\Sigma_{\theta}^{-1}\mu_{\theta}\right)\nonumber \\
 & =\left(\Sigma_{\theta}-\Sigma_{\theta}H^{T}M^{-1}H\Sigma_{\theta}\right)\left(H^{T}\frac{1}{\sigma^{2}}y+\Sigma_{\theta}^{-1}\mu_{\theta}\right)\nonumber \\
 & =\left(I-\Sigma_{\theta}H^{T}M^{-1}H\right)\mu_{\theta}+\left(\Sigma_{\theta}H^{T}\frac{1}{\sigma^{2}}-\Sigma_{\theta}H^{T}M^{-1}H\Sigma_{\theta}\frac{1}{\sigma^{2}}H^{T}\right)y\nonumber \\
 & =\left(I-\Sigma_{\theta}H^{T}M^{-1}H\right)\mu_{\theta}+\frac{1}{\sigma^{2}}\Sigma_{\theta}H^{T}\left(I-M^{-1}H\Sigma_{\theta}H^{T}\right)y\nonumber \\
 & =\left(I-\Sigma_{\theta}H^{T}M^{-1}H\right)\mu_{\theta}+\frac{1}{\sigma^{2}}\Sigma_{\theta}H^{T}M^{-1}\left(M-H\Sigma_{\theta}H^{T}\right)x\nonumber \\
 & =\mu_{\theta}-\Sigma_{\theta}H^{T}M^{-1}H\mu_{\theta}+\frac{1}{\sigma^{2}}\Sigma_{\theta}H^{T}M^{-1}\left(\sigma^{2}I+H\Sigma_{\theta}H^{T}-H\Sigma_{\theta}H^{T}\right)y\nonumber \\
 & =\mu_{\theta}+\Sigma_{\theta}H^{T}M^{-1}\left(y-H\mu_{\theta}\right)
\end{align}
$$

And now we have our equivalent form for the mean of the posterior as well as the covariance:
$$
\begin{align}
\mu_{\theta\mid D} & =\mu_{\theta}+\Sigma_{\theta}H^{T}M^{-1}\left(y-H\mu_{\theta}\right)\\
C_{\theta\mid D} & =\Sigma_{\theta}-\Sigma_{\theta}H^{T}M^{-1}H\Sigma_{\theta}\\
M & = \sigma^2I+H\Sigma_\theta H^T
\end{align}
$$

---


# Regular vs. Equivalent

As discussed (and shown) above, both ways of writing the posterior distributions are valid and equivalent; mathematically, it doesn't matter which of them we use.

However, much like the example for a use of the Woodbury matrix inversion, the sizes of the matrices we need to invert in each of the forms is different, which is what will guide us when we must choose which of them to use. Recall again the covariance in equation  \eqref{eq:post-cov}:

$$
\begin{equation}
C_{\theta\mid D} =\left(\Sigma_{\theta}^{-1}+\frac{1}{\sigma^{2}}H^{T}H\right)^{-1}
\end{equation}
$$
If we have $p$ features, then this matrix will be a $p\times p$ matrix. On the other hand, looking at what we got for the equivalent form:

$$
\begin{equation}
C_{\theta\mid D} =\Sigma_{\theta}-\Sigma_{\theta}H^{T}M^{-1}H\Sigma_{\theta}
\end{equation}
$$
In this form, we need to invert the matrix $M = \sigma^2I+H\Sigma_\theta H^T$ , an $N\times N$ matrix. 

Classically speaking, to ensure that we don't overfit, we would always make sure that $N > p$ . However, using the Bayesian approach we can use more features and still not overfit (as we saw in the last post). The equivalent form becomes useful in the regime where $p > N$ ; when we have more features than data points, since it means that we will have to invert an $N\times N$ matrix<d-footnote>The cost of inverting a $N \times N$ matrix is roughly $O(N^3)$</d-footnote>, which will be more efficient than inverting a $p\times p$ matrix. We will talk about this regime in great detail in later posts.