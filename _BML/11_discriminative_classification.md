---
layout: distill
comments: false
title: Discriminative Classification
description: In this post, we start talking about classification, finally moving on from the world of linear regression. It turns out that exchanging the continuous outputs for discrete ones nullifies all of the maths we saw in the world of linear regression.
date: 2024-03-01
authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University
toc:
  - name: Discriminative Classification
  - name: Classification as Regression (or MSE Classification)
  - name: MAP Estimation
  - name: Discussion
---

<span style="float:left"><a href="https://friedmanroy.github.io/BML/10_gaussian_process/">← Gaussian processes</a></span><span style="float:right"><a href="https://friedmanroy.github.io/BML/12_generative_classification/">Generative classification →</a></span>
<br>
<br>

So far, we have only considered regression in this series of posts. However, regression is not the only task of interest in machine learning and we will now turn our attention towards _classification_. Very broadly, classification is quite similar to regression - in both we get a dataset comprised of pairs $\mathcal{D}=\\{ \left(x\_{i},y\_{i}\right)\\}\_{i=1}^{N}$, and in both we attempt to predict the value of $y_{i}$ given $x_{i}$. However, while in (normal) regression $y_{i}\in\mathbb{R}$ is a scalar, in classification the $y_{i}$s are categorical, i.e. $y_{i}\in\\{ 1,\cdots,C\\}$. This small change ends up affecting and ruining many of the mathematical operations we had up until now, in surprising ways.

Broadly speaking, there are two types of classification: discriminative and generative. We will begin by discussing discriminative classification in this post, wherein we will see that explicitly defining the posterior is basically impossible. We will then move on to generative classification in the next post, which is harder but more forgiving to the Bayesian approach.

# Discriminative Classification

In _discriminative classification_, we attempt to model the distribution of the class label conditional on the input:

$$
\begin{equation}
p\left(y=c\vert x,\theta\right)
\end{equation}
$$

The most common approach to do this is to learn some function $f_{\theta}\left(x;y\right)$ that is supposed to encode the probability that sample $x$ belongs to class $y$. Of course, the probabilities are discrete, and thus must be smaller than 1 while still summing up to 1. To ensure this, the modeled probability is usually defined according to:

$$
\begin{equation}
p\left(y=c\vert x,\theta\right)=\frac{e^{f_{\theta}\left(x;c\right)}}{\sum_{y}e^{f_{\theta}\left(x;y\right)}}
\end{equation}
$$

The exponent ensures that the probabilities are positive, while the sum over exponents ensures that the all probabilities sum to one - this function is sometimes called the _softmax_ function.

In binary classification (when there are only two classes), we can slightly simplify the above. Assuming that $y\in\left\{ 0,1\right\}$, the probability for a class given a sample is modeled as:

$$
\begin{equation}
p\left(y=1\vert x,\theta\right)=\sigma\left(f_{\theta}\left(x\right)\right)\stackrel{\Delta}{=}\frac{1}{1+e^{-f_{\theta}\left(x\right)}}=\frac{e^{f_{\theta}\left(x\right)}}{1+e^{f_{\theta}\left(x\right)}}
\end{equation}
$$

where $\sigma\left(\cdot\right)$ is called the _logistic function_. The logistic function is like a simplified version of the softmax operator we saw above, and has the following useful property:

$$
\begin{align}
p\left(y=0\vert x,\theta\right) & =1-p\left(y=1\vert x,\theta\right)\\
 & =1-\sigma\left(f_{\theta}\left(x\right)\right)\\
 & =1-\frac{1}{1+e^{-f_{\theta}\left(x\right)}}\\
 & =\frac{1+e^{-f_{\theta}\left(x\right)}-1}{1+e^{-f_{\theta}\left(x\right)}}\\
 & =\frac{e^{-f_{\theta}\left(x\right)}}{1+e^{-f_{\theta}\left(x\right)}}=\sigma\left(-f_{\theta}\left(x\right)\right)
\end{align}
$$

In the particular case when probabilities are modeled as a linear function of the parameters, i.e. $f_{\theta}\left(x\right)=h^{T}\left(x\right)\theta$, then the above is called _logistic regression_.

## Associated Likelihood

Notice that above class probabilities describe a sample-specific likelihood. Let's consider the binary case and assume we observe some data $\mathcal{D}=\\{\left(x\_{i},y\_{i}\right)\\}\_{i=1}^{N}$ where $y_{i}\in\\{ 0,1\\}$, then we can write the full likelihood as:

$$
\begin{align}
p\left(\mathcal{D}\vert\theta\right) & =\prod_{i=1}^{N}p\left(y_{i}\vert x_{i},\theta\right)\\
 & =\prod_{i=1}^{N}p\left(1\vert x_{i},\theta\right)^{y_{i}}p\left(0\vert x_{i},\theta\right)^{1-y_{i}}\\
 & =\prod_{i=1}^{N}\sigma\left(f_{\theta}\left(x\right)\right)^{y_{i}}\cdot\left(1-\sigma\left(f_{\theta}\left(x\right)\right)\right)^{1-y_{i}}
\end{align}
$$

So the log-likelihood takes the following form:

$$
\begin{equation}
\log p\left(\mathcal{D}|\theta\right)=\sum_{i=1}^{N}\left[y_{i}\log\sigma\left(f_{\theta}\left(x\right)\right)+\left(1-y_{i}\right)\log\left(1-\sigma\left(f_{\theta}\left(x\right)\right)\right)\right]
\end{equation}
$$

The negative of this term is often called the _binary cross-entropy_
(BCE) loss<d-footnote>This is because the loss is exactly the definition of cross-entropy, if you consider $y_i$ as the true probability for the class label and $\sigma(f_\theta(x))$ as the predicted</d-footnote>.

<br>

## The Problem of Non-Conjugacy 

Notice that this log-likelihood is very un-Gaussian in the parameters $\theta$, unlike every likelihood we have seen so far - even when the function $f_{\theta}$ is linear in the parameters! This is unfortunate as it means that even if we choose a Gaussian prior over the parameters $\theta$, finding the posterior will prove difficult. We might be tempted to try and find a new class of distributions $\mathcal{Q}$ (instead of Gaussians) that might be conjugate to the classification likelihood, i.e.:

$$
\begin{equation}
p\left(\theta\right)\in\mathcal{Q}\Leftrightarrow p\left(\theta\right)p\left(\mathcal{D}|\theta\right)\propto p\left(\theta|\mathcal{D}\right)\in\mathcal{Q}
\end{equation}
$$

However, it turns out that there is no such class of distributions and that there is no closed form solution for the posterior. So it will be quite hard to use the posterior and we will often have to resort to approximations of it in order to make predictions. We will discuss some of these approximations later on in the course in the form of _Markov Chain Monte Carlo_ (MCMC) techniques, although this is still just the tip of the iceberg of approximation techniques.

<br>

# Classification as Regression (or MSE Classification)

As we saw above, the reason the posterior becomes so hard to find is because the likelihood function in classification is not Gaussian. However, we can also use the Gaussian likelihood in order to learn our classifier. In such a case, we will define (as usual):

$$
\begin{equation}
\log p\left(\mathcal{D}|\theta\right)=\sum_{i}\log\mathcal{N}\left(y_{i}|\;f_{\theta}\left(x_{i}\right),\sigma^{2}\right)
\end{equation}
$$

where $\sigma^{2}$ is somehow appropriately chosen to reflect our beliefs about how often the label $y_{i}$ is flipped. Using this likelihood is either called _classification as regression_ or _MSE classification_, since the MSE loss is used in order to train the classifier. In these cases, it is typical to define $y_{i}\in\\{ -1,1\\}$.

This is kind of an "inappropriate likelihood" in the sense that it doesn't truly reflect the way the data was created, but it turns out that this likelihood is also very effective for classification. Given a new point $x^{*}$, our classifier will be:

$$
\begin{equation}
\hat{y}\left(x^{*}\right)=\begin{cases}
1 & f_{\theta}\left(x^{*}\right)\ge0\\
-1 & f_{\theta}\left(x^{*}\right)<0
\end{cases}
\end{equation}
$$

While this method doesn't properly describe our assumptions of the data, it does have a big advantage - it allows us to accurately describe our uncertainty in the prediction. Using the classification rule above, define $p\left(\hat{y}\vert x^{*},\theta\right)$ as the probability of the classification $\hat{y}$ under the posterior. Then, we can find the probability for each class under our model:

$$
\begin{align}
p\left(\hat{y}=-1|x^{*},\theta\right) & =\intop_{-\infty}^{0}p\left(f_{\theta}\left(x^{*}\right)\right)df_{\theta}\left(x^{*}\right)\\
\Leftrightarrow p\left(\hat{y}=1|x^{*},\theta\right) & =1-p\left(y^{*}=-1|x^{*},\theta\right)
\end{align}
$$

<br>

# MAP Estimation

Note that while finding the posterior seems hard, the BCE is frequently optimized using gradient based methods. Because the function $f_{\theta}\left(x\right)$ can usually be efficiently calculated, this makes discriminative classification a very attractive method, since classifying a new point $x_{n}$ involves only one (efficient) calculation. 

As we saw before, when talking about linear regression, we can actually modify the loss used to train the classifier in order to get a MAP estimate. This is done by defining and optimizing the regularized BCE:

$$
\begin{equation}
\log p\left(\theta|\mathcal{D}\right)=\sum_{i=1}^{N}\left[y_{i}\log\sigma\left(f_{\theta}\left(x\right)\right)+\left(1-y_{i}\right)\log\left(1-\sigma\left(f_{\theta}\left(x\right)\right)\right)\right]+\log p\left(\theta\right)+\text{const}
\end{equation}
$$

<br>

# Discussion

This problem of intractability that we have encountered above is surprising, because it doesn't _feel_ like the underlying problem changed much. But this actually happens in many places; for instance, convex optimization is straightforward, but trying to move to integer optimization (which doesn't seem much different) suddenly makes the problem NP-hard.

Because of this, Bayesian discriminative classification usually approximate the posterior or outputs samples from the posterior for classification. Later on, we will talk a bit more about sampling for estimation when the posterior is intractable.

In the next post, we will take a look at _generative classification_, which is somewhat more forgiving to the Bayesian framework but also requires stricter assumptions.

---
<span style="float:left"><a href="https://friedmanroy.github.io/BML/10_gaussian_process/">← Gaussian processes</a></span><span style="float:right"><a href="https://friedmanroy.github.io/BML/12_generative_classification/">Generative classification →</a></span>