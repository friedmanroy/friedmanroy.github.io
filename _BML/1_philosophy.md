---
layout: distill
comments: true
title: The Bayesian Philosophy
description: A high-level description of the frequentist and Bayesian approaches, their differences, and some of their shared qualities.

authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University

toc:
  - name: Frequentist Approach
  - name: Bayesian Approach
  - name: Connections
  - name: Discussion
---

<div style="text-align:right"><a href="https://friedmanroy.github.io/BML/2_estimates/">Estimation and Bayes-Optimal Estimators →</a> </div>

---
<br>

Many tasks in statistics and machine learning can be summarized as attempting to extract information from data. Usually, we will assume that we are given a set of data points:
$$
\begin{equation}
\mathcal{D}=\left\{ x_{1},\cdots,x_{N}\right\} \equiv\left\{ x_{i}\right\} _{i=1}^{N}
\end{equation}
$$
which are assumed to have been drawn<d-footnote>We will usually assume that they are drawn independently from each other but from the same distribute.</d-footnote> from some distribution. Our task, then, is to analyze these data points and extract information from them. 

The driving force of many analyses will be to attempt to find the distribution from which the data points in $\mathcal{D}$ were initially draw. This task is called _modeling_, and (broadly speaking) there are two methodologies to doing so - the frequentist approach and the Bayesian approach.

These two philosophies are often framed as contradictory, but they don't necessarily have to compete with each other. Many times the frequentist problem can be framed using the Bayesian philosophy and vice versa. Understanding both sides of the argument can help build a stronger foundation and intuition for machine learning in general.

In this post, I will attempt to highlight the difference between the approaches, describing the limitations of each of them but also maybe why (philosophically) they are at odds with each other.


<br>
# Frequentist Approach

The frequentist outlook can be further split into two categories - classical and probabilistic. The first outlook is closer to what is regularly taught in intro2ML type classes, while the second is somewhat more structured. The core characteristics of the frequentist philosophy (both probabilistic and not) can be summarized as:

1. There is some true set of parameters $\theta$ which model the data; $\theta$ is **not** a probabilistic object. $\theta$ is assumed to describe a set of hypotheses<d-footnote>We will think of a hypothesis as a possible process for generating data.</d-footnote>, one of which (we believe) was used to generate the data
2. We collect data points $\mathcal{D}$ from this "true model" and want to _estimate_ $\theta$ from these data points using a loss function $L\left(\mathcal{D};\,\theta\right)$

---
All of these definitions tend to be pretty confusing, but hopefully a very simple example will make the meanings behind the terms clearer. 

**Example: Coin Toss**

Suppose we observe the outcomes of coin tosses, where each data point $x$ is either "heads" (denoted by $H$ ), or "tails" (denoted by $T$ ). Such a dataset will look something like:

$$
\begin{equation}
\mathcal{D}=\left\{H,T,T,H,T,H\right\}
\end{equation}
$$

However, we have cause to believe that heads and tails are not equally likely to come up when tossing this coin. Our class of hypotheses will be that there exists some number $\theta\in[0,1]$ such that the above sequence is created according to the following process:

$$
\begin{equation}
p(x_i=H)=\theta=1-p(x_i=T)
\end{equation}
$$

Given the dataset $\mathcal{D}$ , we are tasked with finding the number $\theta$ in some manner. 

How should we go about doing this? The frequentist approach suggests defining some loss function $L(\mathcal{D};\ \theta)$ which will act as a criterion for the quality of our choice of $\theta$ , which we will then attempt to minimize.

---
<br>
## Classical Machine Learning

In the most general form, frequentist machine learning requires two objects, as mentioned before: a set of parameters $\theta$ that define a hypothesis class and a loss function $L\left(\mathcal{D};\,\theta\right)$ . Typically, the loss function is chosen in such a way that it is minimized by the wanted outcome, although this can be hard to control in many real life applications. 

Framing the problem of machine learning in this manner results in a deterministic algorithm, whose correctness is then left to be proven. Examples of such algorithms are _decision trees, support vector machines_ (SVMs) and _k-means_, whose definition and solution are inherently not probabilistic. 

---

**Example: Coin Toss (again)**
Through a simple thought experiment, we can convince ourselves that for any value of $\theta$ , if we look at a dataset of size $N$ and count the number of heads $N_H$ in said dataset, then at the limit of very large $N$ we will usually observe:

$$
\begin{equation}
\frac{N_H}{N}\stackrel{N\rightarrow\infty}{=}\theta
\end{equation}
$$

If this is true, then maybe a loss of the following form makes sense:

$$
\begin{equation}
L(\mathcal{D};\ \theta)=\left\vert\theta-\frac{N_H}{N}\right\vert
\end{equation}
$$

The value of $\theta$ that minimizes this loss is, rather intuitively, $\hat{\theta}=\frac{N_H}{N}$ and so we will say that this is our guess for the value of $\theta$ (which is why I added the $\hat{}$ , to remind us that this isn't the true value, only our guess). This estimate has the nice property of being correct in the limit of infinite data (if we believe that this is truly the process that generated the data):

$$
\hat{\theta}\stackrel{N\rightarrow\infty}{=}\theta
$$

This example is simplified to the point of being cartoonish, but this a very simplified sketch of one method for estimation.

---
<br>
## Probabilistic Machine Learning

This form of frequentist machine learning is slightly more structured, framing the problem probabilistically. In this form, a stochastic model for the generation of the data is assumed. This model has parameters $\theta$ and a distribution attached to these parameters $p\left(\mathcal{D};\,\theta\right)$ which controls how likely it is for us to have observed the data under a specific choice of the parameters $\theta$ . Here, while $\theta$ appears in a density function, it is _not a probabilistic object_; that is what the semi-colon (the ";" sign) is meant to convey. The solution to this problem is then to find the parameters $\theta$ that created the data points in $\mathcal{D}$ .

Many times, the classical and probabilistic views are connected. Many classical algorithms can be reframed in a probabilistic framework and vice versa. However, the probabilistic outlook allows us to explicitly take into account the stochastic nature of the data, which allows for a more careful way to both _define_ and _solve_ problems.

### Maximum Likelihood Estimation

One of the most common probabilistic criterions used to estimate the parameters is called the likelihood. Given a data set $\mathcal{D}$ , we define the likelihood as:
$$
\begin{equation}
L\left(\theta\right)\stackrel{\Delta}{=} p\left(\mathcal{D}\;;\theta\right)=\prod_{i=1}^{N}p\left(x_{i}\;;\theta\right)
\end{equation}
$$
Notice, of course, that the likelihood is a function of the parameters $\theta$ . In this definition, we used the fact that the points were drawn _i.i.d._ from $p\left(\cdot\;;\theta\right)$ in order to multiply their probabilities - if they weren't drawn independently, we couldn't have done this!

Having defined this criterion, the natural step in order to estimate the parameters $\theta$ is to maximize the likelihood:

$$
\begin{equation}
\hat{\theta}_{ML}\stackrel{\Delta}{=}\arg\max_{\theta}L\left(\theta\right)
\end{equation}
$$

after all, if $\theta$ maximizes the likelihood, it is the most likely set of parameters to describe the distribution<d-footnote>We will actually show a more formal reason to use MLE in the <a href="https://friedmanroy.github.io/BML/2_estimates/">next post about estimation</a>.</d-footnote>. This estimate is called the _maximum likelihood estimate_ (MLE) of the distribution and we will denote it by  $\hat{\theta}\_{ML}$  (the  $\hat{}$  is to remember that it is an estimate and the  $\_{ML}$  is to remember that it maximizes the likelihood). 

Also, usually the log-likelihood is maximized instead of the likelihood, defined as:
$$
\begin{equation}
\ell\left(\theta\right)\stackrel{\Delta}{=}\log L\left(\theta\right)=\sum_{i=1}^{N}\log p\left(x_{i}\;;\theta\right)
\end{equation}
$$
The result is the same (since the logarithm is a strictly monotonically increasing function), however this includes maximizing a sum instead of a product, which is usually easier.

---
#### Example: Coin Toss (yes, again)

Coming back to our coin toss example, if we assume that each coin toss is independent of the one that came before it, then the likelihood that we saw $\mathcal{D}$ is given by:

$$
\begin{align}
p(\mathcal{D};\ \theta)&=\prod_i p(x_i;\ \theta)\\
&=\prod_{i:\ x_i=H}p(x_i=H)\prod_{j:\ x_j=T}p(x_i=T)\\
&=\prod_{i:\ x_i=H}\theta\prod_{j:\ x_j=T}(1-\theta)\\
&=\theta^{N_H}\cdot(1-\theta)^{N-N_H}
\end{align}
$$

Taking the log of this expression, we get:

$$
\begin{equation}
\log p(\mathcal{D}; \theta)=N_H\log\theta +\left(N-N_H\right)\log(1-\theta)
\end{equation}
$$

To find that maximum of this log-likelihood, we can differentiate and equate to 0. After a bit of (simple) math, we will get:

$$
\begin{align}
\frac{\partial \log p(\mathcal{D};\theta)}{\partial \theta }&=\frac{N_H}{\theta}-\frac{N-N_H}{1-\theta}\stackrel{!}{=}0\\
\Leftrightarrow N_H(1-\theta) &=(N-N_H)\theta\\
\Leftrightarrow N\theta &=N_H\\
\Leftrightarrow \hat{\theta} &=\frac{N_H}{N}
\end{align}
$$

and, amazingly, we got the same estimator as the previous approach!

---
<br>
# Bayesian Approach

The Bayesian philosophy assumes that we have some knowledge about the distribution the points were drawn from ahead of time, i.e. we assume that the parameters themselves have some distribution $p\left(\theta\right)$ . This distribution is usually called the _prior distribution_, because we assume we have some prior knowledge. This means that there is no single true value for $\theta$ , rather that a distribution of $\theta$s could have given rise to the data. That is, unlike the frequentist view where $\theta$ _is by definition not probabilistic_, under the Bayesian view we assume that there is some distribution over $\theta$ s.

In this new outlook, instead of trying to find the $\theta$ that generated the data, we will try to update our knowledge regarding which values $\theta$ could have had to create the data. We will want to find is the _posterior distribution_ $p\left(\theta\mid\mathcal{D}\right)$ , so called because we update our beliefs _after_ the fact (in Latin "post" means "after", while "prior" means "before"). Using Bayes' law, we can describe this using the prior and likelihood distributions:
$$
\begin{equation}
\overbrace{p\left(\theta\mid\mathcal{D}\right)}^{\text{posterior}}=\frac{\overbrace{p\left(\mathcal{D}\mid\theta\right)}^{\text{likelihood}}\overbrace{p\left(\theta\right)}^{\text{prior}}}{p\left(\mathcal{D}\right)}\propto\overbrace{p\left(\mathcal{D}\mid\theta\right)}^{\text{likelihood}}\overbrace{p\left(\theta\right)}^{\text{prior}}
\end{equation}
$$
Usually we assume that the data set is held constant, so $p\left(\mathcal{D}\right)$ does not affect the calculation of the posterior probability, which is why it is usually disregarded (or swallowed up by the $\propto$ sign). The likelihood term here $p\left(\mathcal{D}\mid\theta\right)$ is actually exactly the same as the frequentist likelihood $p\left(\mathcal{D}\;;\theta\right)$ , only now we can properly condition on $\theta$ . 

As mentioned, the posterior distribution is an updated version of our beliefs, and gives a new distribution over which values of $\theta$ are likely. That said, we can also extract point estimates (single estimates) of $\theta$ from the posterior. For example:

1. The _maximum a-posteriori_ (MAP) estimate is defined as: 
$$
\hat{\theta}_{MAP}\stackrel{\Delta}{=}\arg\max_{\theta}p\left(\theta\mid\mathcal{D}\right)=\arg\max_{\theta}p\left(\theta\right)p\left(\mathcal{D}\mid\theta\right)
$$
2. The _minimum mean squared error_ (MMSE) estimate is defined as: $\hat{\theta}_{MMSE}\stackrel{\Delta}{=}\mathbb{E}\left[\theta\mid\mathcal{D}\right]$ . As the name suggests, the MMSE is the optimal estimator under a mean squared error loss (assuming our prior is correct)

Finally, many times we are not interested in the posterior over parameter values $\theta$ , but actually only care about the predictions. In this case, we can define the _posterior predictive distribution_ (PPD) defined as:
$$
\begin{equation}
p\left(d^{*}\mid\ \mathcal{D}\right)\stackrel{\Delta}{=}\intop p\left(d^{*}\mid\ \theta\right)p\left(\theta\mid\,\mathcal{D}\right)d\theta
\end{equation}
$$
where $d^{*}$ is a newly observed data point.

---

#### Example: Coin Toss (last time, promise)

The only real difference in this example from the MLE version is that now we want to weight each outcome by the prior distribution. The prior distribution conveys our beliefs on the outcomes, which is why it is sometimes called a *subjective* probability.

For this example, having heard about this experiment and seeing it conducted many times in the past, my belief is that it is more likely that the coin is a fair coin, i.e. $\theta\approx 1/2$ than otherwise. However, I additionally feel that it is not impossible for the coin to be unfair (in either direction), so the prior chosen according to previously experiments I've seen in the past is:

$$
\begin{equation}
p(\theta)=\frac{1}{2.5}\cdot\cases{4 & $\theta\in[1/4,3/4]$ \\ 1 & otherwise }
\end{equation}
$$

That is, it is 4 times more likely for $\theta$ to be in the range $[1/4,\ 3/4]$ than outside it. Using this prior and a dataset, we can now calculate the posterior distribution which will effectively act as an updated probability for possible values of $\theta$ .

---
<br>
### Appeal of the Bayesian Approach

Similarly to probabilistic ML, Bayesian ML allows us to elegantly define our problems and how to solve them. All we need to do is define the likelihood function, our prior, and we have the solution - find the posterior distribution. This counters that of the classical ML approach, where a loss function and algorithm have to be found such that: (a) minimizing the loss gives a solution to the problem and (b) the algorithm actually minimizes the loss. 

Additionally, the Bayesian approach allows us to inject prior knowledge of the problem in order to generate a solution through the density $p\left(\theta\right)$ . This prior knowledge can be very helpful if not much data has been observed. Finally, under certain circumstances, it can be proved that the Bayesian approach will give the _optimal_ solutions.

### Problems with the Bayesian Approach

The biggest flaw (and strength!) of Bayesian ML is the prior distribution $p\left(\theta\right)$ . Since the prior can _heavily_ affect the solution, it is imperative that it is chosen to correctly reflect the space of solutions. But many times we have no prior knowledge over solutions; for instance, how can we know what the distribution over the parameters of a neural network should look like? This is a very ambiguous decision that has to be made, and many times it is not obvious how to make it. In practice, this means the prior is often arbitrarily chosen and doesn't necessarily reflect the distribution over parameters we want. 

An additional difficulty of Bayesian methods is the actual calculation of the posterior $p\left(\theta\mid\,\mathcal{D}\right)$ . While the learning "algorithm" in Bayesian ML is conceptually straightforward, many times the posterior is intractable. In which case approximations or point estimates must be used. These approximations, many times, are based on MCMC samples from the posterior or variational methods, whose quality is difficult to verify.

However, both of the above flaws are usually shared with methods in classic ML. Typically, the loss used in such methods is heuristically chosen - much like the prior in Bayesian approaches. Further, while it can be shown that the solution is reached when the loss is minimized, it is usually difficult (if not impossible) to show that the algorithm has actually reached a minimum.


<br>
# Connections

If we assume an uninformative prior over $\theta$ , i.e. all values of $\theta$ are equally probable and the prior doesn't add any knowledge as to the choice of $\theta$ :
$$
\begin{equation}
p\left(\theta\right)\propto1
\end{equation}
$$
then in this case the MAP estimate is:
$$
\begin{align}
\hat{\theta}_{MAP} & =\arg\max_{\theta}p\left(\theta\right)p\left(\mathcal{D}\mid\theta\right)\nonumber \\
 & =\arg\max_{\theta}p\left(\mathcal{D}\mid\theta\right)\times\text{const}\nonumber \\
 & =\arg\max_{\theta}p\left(\mathcal{D}\mid\theta\right)=\hat{\theta}_{ML}
\end{align}
$$
So we see that the frequentist estimate $\hat{\theta}\_{ML}$ is a special case of the Bayesian estimate  $\hat{\theta}\_{MAP}$ !

Actually, we did something extremely fishy when we said that $p\left(\theta\right)$ is "uniform" - this isn't possible in many cases! A uniform probability over all of the real line $\mathbb{R}$ is impossible... so how can we even talk about this so called "uninformative prior"? While this is true, as long as the posterior $p\left(\theta\mid\mathcal{D}\right)$ is well defined, the MAP and MMSE estimates will still exist. In this special case, the frequentist and Bayesian world views collide, and it will be useful to keep this fact in mind.

<br>
# Discussion

Having shown the two possible philosophies, frequentist and Bayesian, it is important to keep in mind that both have viable methods and shortcomings. As a consequence, throughout this series of posts we will typically consider both of these approaches together. Usually, the Bayesian approach will _augment_ the frequentist approach, giving us some structured method to inject prior knowledge into our predictions.

So far, everything was very abstract and amorphous. Starting with the next post, we'll go into some more details. Specifically, we will see how (and when) using estimators extracted using the Bayesian approach will typically be optimal. This necessitates the definition of "optimality" in estimators, which we will build using decision theory.

<br>

---
<div style="text-align:right"><a href="https://friedmanroy.github.io/BML/2_estimates/">Estimation and Bayes-Optimal Estimators →</a> </div>