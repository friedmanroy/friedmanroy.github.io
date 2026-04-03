---
layout: distill
comments: false
title: Gibbs Sampling
description: The most common approaches to sampling from complex distributions are Monte Carlo Markov Chain (MCMC) algorithms. In this post we explore the simplest MCMC algorithm, called Gibbs sampling.
date: 2026-04-01
authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University
toc:
  - name: Motivation
  - name: Markov Chain Monte Carlo
  - name: Gibbs Sampling
  - name: Back to  Robust Regression
  - name: Discussion
---

<span style="float:left"><a href="https://friedmanroy.github.io/BML/14_sampling/">← Benefits of sampling</a></span><span style="float:right"><a href=""> →</a></span>
<br>
<br>

In most practical scenarios when using probabilistic models, exact inference involving the posterior distribution is either very computationally expensive or outright impossible. Instead, it is common to resort to some form of an approximation. A very common method for approximate inference is using samples from the distribution in order to estimate some statistics, and the simplest of these methods is called Gibbs sampling.

---
# Motivation

> Disclaimer: this part of the post only makes sense if you read the previous posts about [linear regression](https://friedmanroy.github.io/BML/5_linear_regression/), [Gaussian mixture models](https://friedmanroy.github.io/BML/13_gmms/) and [posterior sampling](https://friedmanroy.github.io/BML/14_sampling/). But, it's not really required to understand the concept of Gibbs sampling, so you can skip directly to the [Markov Chain Monte Carlo](https://friedmanroy.github.io/BML/15_gibbs/#markov-chain-monte-carlo) section and skip this motivation if you didn't read those other posts.

Gibbs sampling, when you first hear about it, seems like a very weird algorithm that doesn't necessarily make sense. So, before we begin, I want to give some motivation and build some intuition for why we need Gibbs sampling, and how it works.

Let's consider a new form of linear regression (yes, there are a million types of linear regression, it seems) - _robust regression_ or outlier regression. In robust regression, we assume that a few of the data points are outliers, and are essentially drawn from some noise distribution:

$$
\begin{equation}
y=\begin{cases}\theta^T h(x)+\epsilon&\text{w.p.}\ \pi_\text{in}\\ M\epsilon&\text{w.p.}\ 1-\pi_\text{in}\end{cases}
\end{equation}
$$

where $\epsilon\sim\mathcal{N}(0,\sigma^2)$, $M\gg 1$ and $0<\pi_\text{in}<1$ is the marginal probability that any data point is an inlier. So, usually something like $\pi_\text{in}=0.95$ is assumed, meaning around 5% of the data might be outliers. For our purposes, we will (of course) assume that we have a Gaussian prior on $\theta$.

In practice this is like a GMM likelihood on top of a Gaussian prior. If you do the math, this turns into a GMM posterior. In particular, for *every data point* the likelihood has two components, because we need to consider the case when it is an inlier *and also* when it is an outlier. So, the posterior will be a GMM with $2^N$ centers. Because of this, even for moderate amounts of data, it becomes infeasible to keep the whole posterior in memory.

{% details MLE for outlier regression %}
Before we start finding ways to do approximate Bayesian inference on this problem, it's worth considering the MLE solution, to see whether it has a direct solution.

Let's start by writing the log-likelihood of the problem:
$$
\begin{align}
\log p(\mathcal{D}\vert\theta)=\sum_{i=1}^N \log\left[\pi_\text{in}\cdot \mathcal{N}\left(y_i\vert\ \theta^Th(x_i),I\sigma^2\right)+(1-\pi_\text{in})\cdot\mathcal{N}\left(y_i\vert\ 0,I\sigma^2_\text{outlier}\right)\right]
\end{align}
$$
This optimization problem is very similar to the one we saw for [GMMs](https://friedmanroy.github.io/BML/13_gmms/). Indeed, the easiest way to optimize the likelihood is through an EM algorithm like we saw in the [GMMs](https://friedmanroy.github.io/BML/13_gmms/) post. There, we had a hidden (categorial) random variable $z_i$ which determined the attribution of each data point $x_i$ to each of the centers of the GMM. Here, $z_i$ will indicate whether $y_i$ is an inlier or outlier, and the stages of the EM algorithm will be: (E-step) for each data point, calculate the probability that it is an inlier, (M-step) a weighted least squares problem. You can try to go through the derivation itself, the E- and M-steps should get will look something like:

$$
\begin{aligned}
\text{(E-step):}\qquad& \forall i\quad z_i(\theta_{(t)})=\frac{\pi_\text{in}\cdot \mathcal{N}\left(y_i\vert\ \theta_{(t)}^Th(x_i),I\sigma^2\right)}{\pi_\text{in}\cdot \mathcal{N}\left(y_i\vert\ \theta_{(t)}^Th(x_i),I\sigma^2\right)+(1-\pi_\text{in})\cdot\mathcal{N}\left(y_i\vert\ 0,I\sigma^2_\text{outlier}\right)}\\ 
\text{(M-step):}\qquad& \theta_{(t+1)}=\arg\min_\theta\sum_{i=1}^N z_i(\theta_{(t)})\cdot\|\theta^Th(x_i)-y_i\|^2
\end{aligned}
$$

A point that is a bit strange about this result, is that points can be chosen as partially inliers and outliers, at the same time. This is because $z_i(\theta_{(t)})$ represents the _mean_ probability that data point $i$ is an inlier.

Using Gibbs sampling, estimating the robust regression parameters is actually quite similar to this EM algorithm, with two key distinctions: (1) points are chosen either as inliers or outliers, not somewhere in between; and (2) at each stage, the parameters $\theta_{(t+1)}$ are sampled from the posterior.

{% enddetails %}

## The Benefits of Sampling, Once Again

Instead, what if we had an efficient way to sample from the posterior, without actually keeping all of it in memory? Here, a sample from the posterior is straightforward to keep in memory: for each point, we only need to know whether it was sampled as an inlier or outlier, so that's a list of $N$ 0s (for inliers) and 1s (for outliers). This effectively chooses which Gaussian in the GMM needs to be used to sample the weights $\theta$. Each sample then gives us a _guess_ for the true behavior, which is informed by our prior. 

So, if we have some magical way to sample assignments to inliers/outliers, and the regression parameters, from the posterior, then we bypass the problem of keeping the whole posterior in memory and we can make predictions directly using these samples instead. This method, where we sample from the posterior to make predictions, is often called Monte Carlo sampling (better definition down below, don't worry).

## Starting Rough and Refining

Realistically, sampling directly from the posterior is not easy to do. Only in very specific cases (such as our Gaussian posterior example) is it possible to find a simple way to sample from the posterior. Instead, many times an iterative approach is taken, where we start with an initial _estimate_ for a sample, and then (usually, slowly) refine it, making sure the refinement can only make the sample more plausibly having been drawn from the posterior. Processes like this are called Markov chain Monte Carlo (MCMC) methods, and this post is all about a specific MCMC algorithm called Gibbs sampling.

For the outlier regression above, Gibbs sampling results in an intuitive iterative algorithm. First, we randomly guess which points are inliers and which are outliers, making sure that the probability for an outlier is still $1-\pi_\text{in}$. Then, having made these choices, the algorithm fits the posterior of a linear regression problem _while ignoring all outliers_. This gives the first estimate for the regression, $\theta_1$, and is the basis for the next round. Using this guess, it is possible to re-evaluate all data-points and ask: for all data points, what is the probability that they are actually outliers, under the choice of parameters $\theta_1$? Each point is then sampled to be considered as an outlier according to this probability, giving us a new list which dictates which data points should be considered as inliers and which as outliers. Then, a new set of parameters is sampled, $\theta_2$, from the posterior of a linear regression model that ignores the outliers. And this goes on and on, until we decide we had enough and return the last set of parameters $\theta_T$ after $T$ iterations.

In simpler terms - at each iteration, we guess which points are inliers and outliers based on the quality of the linear regression, and then fit the linear regression ignoring the outliers. Because there are few outliers, and they are usually far from the rest of the data, the linear regressor has a hard time fitting them. At each iteration of the algorithm, this pushes outliers (i.e. difficult points) to be ignored, resulting in a more robust regression. I'll give the exact algorithm after properly presenting the Gibbs sampling algorithm.

The amazing thing about this algorithm is that if we go through the iterative process enough times we can be sure that the sampled weights $\theta_T$ are from the distribution of the true posterior.

---
# Markov Chain Monte Carlo

Before actually diving into Gibbs sampling, we'll define the wider family of techniques called Markov chain Monte Carlo (MCMC) methods. As you can see, there are two “MC”s in MCMC, so let's start by defining each of them.

## Markov Chains

> **Definition - Markov Chain:** A Markov chain is composed of an ordered set of random variables (possibly infinite) $x_{1},\cdots,x_{T}\in\mathbb{R}^{d}$, called a “chain”. A chain is called a Markov chain if it satisfies the following Markov assumption:
> 
> $$
  \begin{equation}
  \forall t\quad p\left(x_{t}\vert x_{1},\cdots,x_{t-1}\right)=p\left(x_{t}\vert\ x_{t-1}\right)
  \end{equation}
  $$
>  

We can think of a Markov chain as a random walk, where the probability to reach some point $x_{t}$ depends only on where the walker was in the previous state (which is why Markov chains are also sometimes called “memory-less”). 

An important attribute of Markov chains is that if you simulate a random walk under their "update rule" $p(x_t\vert x_{t-1})$, then no matter what the value of $x_1$,  $x_T$ is guaranteed to be a sample from the _stationary distribution_ of the Markov chain when $T$ is very large (and under some conditions). 

> **Definition - Stationary Distribution:** Let $x_{1},x_{2},x_{3},\cdots$ be a Markov chain. The distribution $\pi\left(x\right)$ over the values of each random variable in the chain is called the stationary distribution of the Markov chain if:
> 
$$
\begin{equation}
\forall t\quad\sum_{x_{t-1}}\pi\left(x_{t-1}\right)p\left(x_{t}\vert\ x_{t-1}\right)=\pi\left(x_{t}\right)
\end{equation}
$$
> 
> That is, if a random variable $x_{t-1}$ is a sample from the stationary distribution of the Markov chain, then the distribution of the next random variable $x_{t}$ will also be the stationary distribution. Stationary distributions of Markov chains are not necessarily unique. However, if the Markov chain has a unique stationary distribution, then:
> 
$$
\begin{equation}
p\left(x_{T}\right)\stackrel{T\rightarrow\infty}{=}\pi\left(x\right)
\end{equation}
$$

Not all Markov chains have a stationary distribution, and those that do might not have a single, unique stationary distribution. However, it is enough for the Markov chain to be _ergodic_ for the chain to have a unique stationary distribution:

> **Definition - Ergodic Markov Chain:** Let $x_{1},x_{2},x_{3},\cdots$ be a Markov chain. If for every two points $x$ and $x'$ there exists $N\in\mathbb{N}$ such that:
> 
$$
\begin{equation}
\forall t\quad\forall n\ge N\quad p\left(x_{t+n}=x'\,\vert\ \,x_{t}=x\right)>0
\end{equation}
$$
> 
> then the Markov chain is called ergodic. In other words, a Markov chain is ergodic if we can reach any state from any other state, after enough iterations (after which the probability is always positive). If a Markov chain is ergodic, then it has a stationary distribution and it is unique.


Okay, that's a lot of definitions... However, most of them, while looking kind of weird, should be pretty intuitive. Markov chains are frequently used for many ML tasks, whether Bayesian or not, so it's a good idea to get comfortable with their definitions.


## Monte Carlo

[Monte Carlo (MC) methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) are a very broad class of computational algorithms which can basically be described as methods for the estimation of deterministic quantities using stochastic samples. 

For instance, say we have some complicated distribution $p(x)$ and a function $f(x)$. To calculate the mean of $f(x)$ under $p(x)$ requires us to solve the following integral:
$$
\begin{equation}
\mathbb{E}_x[f(x)]=\intop f(x)p(x)dx
\end{equation}
$$
which is often either very difficult or intractable (meaning there is no closed-form solution). But, suppose we know how to sample points $x_i$ from $p(x)$:
$$
\begin{equation}
x_{1},\cdots,x_{N}\underset{\text{i.i.d.}}{\sim}p\left(x\right)
\end{equation}
$$
Then, we can approximate the above integral only using our samples:
$$
\begin{equation}
\mathbb{E}_x\left[f(x)\right]\stackrel{N\rightarrow\infty}{=}\frac{1}{N}\sum_{i=1}^{N}f(x_{i})
\end{equation}
$$
The quality of the approximation depends on how many i.i.d. samples $x_i$ we have from $p(x)$, typically requiring a large amount of points to converge to an accurate approximation. But, this gives us a way to turn _stochastic_ observations into approximations of _deterministic_ values.


## Markov Chain Monte Carlo

MCMC methods are basically Monte Carlo techniques utilizing Markov chains. In particular, say we want to calculate something with regards to some distribution $p\left(x\right)$, but we don't know its analytical form or how to sample from it directly. Instead, in MCMC methods, what we do have is some Markov chain $x_{1},\cdots,x_{T}$ such that:

$$\begin{equation}
x_{T}\stackrel{T\rightarrow\infty}{\sim}p\left(x\right)
\end{equation}
$$

In other words, we have a Markov chain whose stationary distribution is the distribution we need for our Monte Carlo algorithm.

---
# Gibbs Sampling

Gibbs sampling is an MCMC method for sampling from general distributions, when we know how to sample from conditional distributions. The main idea behind Gibbs sampling is to update each variable iteratively by sampling it conditioned on all of the other variables. 

Suppose we want to sample from some distribution:

$$
\begin{equation}
p\left(x\right)=p\left(x_{1},\cdots,x_{d}\right)
\end{equation}
$$

and don't know how to sample from $p\left(x\right)$ directly, but do know how to sample from:

$$
\begin{equation}
p\left(x_{i}\vert\ x_{-i}\right)\stackrel{\Delta}{=} p\left(x_{i}\vert\ x_{1},\cdots,x_{i-1},x_{i+1},\cdots,x_{d}\right)
\end{equation}
$$

for any index $i$. Then, Gibbs sampling is the iterative algorithm that runs through all indices, sampling the corresponding random variable conditioned on all the other variables. The algorithm can be summarized as:

1. Initialize $x^{\left(0\right)}=\left(x_{1}^{\left(0\right)},\cdots,x_{d}^{\left(0\right)}\right)^{T}$ in some manner
2. For $t=1\ldots T$:
	1. For $i=1\ldots d$:
		Sample $x_{i}^{\left(t\right)}\sim p\left(x_{i}\vert\ x_{1}^{\left(t\right)},\cdots,x_{i-1}^{\left(t\right)},x_{i+1}^{\left(t-1\right)},\cdots,x_{d}^{\left(t-1\right)}\right)$

To ease notations a bit, we will use the notation $x_{-i}=\left\{ x_{1},\cdots,x_{i-1},x_{i+1},\cdots,x_{d}\right\}$ , so that $x_{-i}$ contains all of the variables except $x_{i}$. 

So, the Gibbs sampler for a joint distribution $p\left(x_{1},\cdots,x_{d}\right)$ is completely defined by all of the conditional distributions $p\left(x_{i}\vert\ x_{-i}\right)$.  So, we don't even need to know the exact form of the full distribution in order to sample, only conditionals from the distribution! 

This sounds like a very synthetic scenario, where we have access to all of $p\left(x_{i}\vert\ x_{-i}\right)$ but not the joint. However, note that it is exactly the setting of robust regression from the start of the post. There, the distribution we want to sample from is $p\left(\theta,z_1,\cdots,z_N\vert\ \mathcal{D}\right)$ where $\theta$ are the regression parameters and $z_i\in\{0,1\}$ is a binary random variable which dictates if data point $i$ is an inlier ($z_i=0$) or an outlier ($z_i=1$). Note that, in that scenario, sampling $\theta$ given all the $z_i$s is simple (linear regression ignoring all outliers), and each $z_i$ is independent of $z_j$ once we have access to $\theta$. So, while it's difficult to sample from the full posterior, it's easy to sample from each of the conditional distributions, making Gibbs sampling particularly effective.

### Simple Example: Bivariate Gaussian

Suppose we want to sample from the bivariate Gaussian distribution:

$$
\begin{equation}
p\left(x,y\right)=\mathcal{N}\left(\left(\begin{matrix}x\\
y
\end{matrix}\right)\vert\ \mu, \Sigma \right)
\end{equation}
$$

with $\mu=\left(\mu_{x},\mu_{y}\right)^{T}$ and
$$
\Sigma=\left(\begin{matrix}\sigma^2_{x} & \sigma_{xy}\\ \sigma_{xy} & \sigma^2_{y} \end{matrix}\right)
$$. 

We have already [worked out the conditional distributions for this Gaussian in the past](https://friedmanroy.github.io/BML/3_gaussians/):

$$
\begin{align}
p\left(x\vert\ y\right)	&= \mathcal{N}\left(x\vert\ \mu_{x}-\frac{\sigma_{xy}^{2}}{\sigma^2_{y}}\left(y-\mu_{y}\right),\sigma^2_{x}-\frac{\sigma_{xy}^{2}}{\sigma^2_{y}}\right) \\
p\left(y\vert\ x\right)	&= \mathcal{N}\left(y\vert\ \mu_{y}-\frac{\sigma_{xy}^{2}}{\sigma^2_{x}}\left(x-\mu_{x}\right), \sigma^2_{y}-\frac{\sigma_{xy}^{2}}{\sigma^2_{x}}\right)
\end{align}
$$

The Gibbs sampling algorithm simply draws $x$ and $y$ iteratively from the two distributions in the equations above. The procedure itself, following a single chain, is represented by the following animation:

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/single_gibbs.gif"  
alt="Gibbs sampling from a bivariate Gaussian distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 1: Gibbs sampling from a bivariate Gaussian. The position of the black dot represents the position at each iteration of the Gibbs sampling algorithm. Note how the sampling process updates one dimension at a time - first only the horizontal position is updated, and then only the vertical. This corresponds to sampling from the conditional distribution $p(x\vert y)$ and then $p(y\vert x)$.
</div>


The bivariate Gaussian is also a good representation for problems related to the Gibbs sampling procedure. Let's consider the scenario where $\sigma^2\_{x}=\sigma^2\_{y}=L$ and $\ell=L-\frac{\sigma_{xy}^{2}}{L}$ such that $\ell\ll L$. In this case, the variance of each of the marginal distributions $p\left(x\right)$ and $p\left(y\right)$ is $L$, while the variance of the conditional distributions $p\left(x\vert\ y\right)$ and $p\left(y\vert\ x\right)$ is $\ell$. Since $\ell$ is much smaller than $L$, the number of steps needed for two iterations of the sampling procedure to move a large distance will be a function of the ratio of the variances $L/\ell$. The more correlated the variables $x$ and $y$ will be, the smaller $\ell$ becomes, which means more steps must be taken for samples from the Markov chain to be far apart. This phenomenon can be clearly seen in the following animation:

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/single_gibbs_99corr.gif"  
alt="Gibbs sampling from a highly-correlated bivariate Gaussian distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 2: Gibbs sampling from a highly correlated bivariate Gaussian. Because $x$ and $y$ are highly correlated, each coordinate-wise step remains close to the previous iteration. This, in turn, is related to the speed with which the Gibbs sampling algorithm converges to the Markov chain's stationary distribution.
</div>

Scenarios such as these, where we need to wait a long time for the Markov chain to converge to the stationary distribution, is one of the most difficult problems in MCMC methods. To understand why, we need to shift our attention to how any MCMC method is used. 
Using MCMC, it is common to start with some random initial position $x^{(0)}$, which is not a typical sample from the stationary distribution of the Markov chain, $p(x)$. When the joint distribution $p(x)$ is highly correlated, as in the example above, that means we will need to run the sampling algorithm for a long time $T\rightarrow\infty$ before we can be sure that $x^{(T)}\sim p(x)$. In turn, this means that the computational load due to the MCMC calculation will be quite large if many independent samples from the distribution are needed to get an accurate estimate. Below I go into further details regarding these limitations and considerations. 

## Sampling using Markov Chains

Gibbs sampling and other MCMC methods are guaranteed to work when the Markov chain is ergodic. If it is, and we run the Gibbs sampler $N$ times for $T$ iterations, resulting in the variables $z_{1}^{\left(T\right)},\cdots,z_{N}^{\left(T\right)}$, then we can confidently say:

$$
\begin{equation}
z_{1}^{\left(T\right)},\cdots,z_{N}^{\left(T\right)}\stackrel{\text{i.i.d.}\ ,\ T\rightarrow\infty}{\sim}p\left(x\right)
\end{equation}
$$

That is, running the Gibbs sampler $N$ independent times with a large number of iterations $T$, the outputs of each call to the Gibbs sampler will be an i.i.d. sample from the distribution we want to sample from. Notice that this is only guaranteed when $T$ is sufficiently large! 

Alternatively, assume that we run the Gibbs sampler only once for $T+\Delta$ iterations and the outputs are the variables at each step of the algorithm, $z^{\left(1\right)},\cdots,z^{\left(T\right)},\cdots,z^{\left(T+\Delta\right)}$. In this case, which is completely equivalent to what we saw before, we can confidently say:

$$
\begin{align}
z^{\left(T\right)},z^{\left(T+\Delta\right)}	&\stackrel{T\rightarrow\infty}{\sim}p\left(x\right)\\
z^{\left(T\right)},z^{\left(T+\Delta\right)}	&\stackrel{\text{i.i.d.}\ ,\ \Delta,T\rightarrow\infty}{\sim}p\left(x\right)
\end{align}
$$

In other words, if we wait enough time on the chain, the consequent samples from the chain will be from the distribution. If we wait enough time between these samples, then they will also be i.i.d. (this is like saying we started the chain at $z^{\left(T\right)}$ and waited $\Delta$ iterations).

## Mixing Time

As written above, we need to wait a while after initializing the chain before the samples are provably drawn from the distribution we actually want to sample from. But how long do we actually have to wait in practice? The mixing time is a formal definition for this duration, and has many forms, so we will try to give a general definition here.

> **Definition - Mixing Time:** Let $x_{1},\cdots,x_{T}$ be a Markov chain with a (unique) stationary distribution $\pi\left(x\right)$ and let $\mathcal{D}\left(\mathcal{P},\mathcal{Q}\right)$ be some measure for the distance between the distributions $\mathcal{P}$ and $\mathcal{Q}$. In addition, suppose that we have some way of initializing the Markov chain, that is we have some distribution $q\left(x\right)$ that dictates a distribution over the possible initial states. We will define $p_{q}^{\left(T\right)}\left(x\right)$ as the distribution that is reached after T steps on the Markov chain, if starting from the distribution $q\left(x\right)$. Then the $\epsilon$-*mixing time* of the Markov chain is defined as:
> 
> $$
\Delta_{\mathcal{D}}^{\left(\epsilon\right)}=\arg\min_{T}\left\{ T\in\mathbb{N}\:\vert\ \:\mathcal{D}\left(p_{q}^{\left(T\right)},\pi\right)\le\epsilon\right\} 
$$

The definition above is only a (very) rough definition of what is used in the literature, but will be good enough for us. Intuitively, the mixing time is the number of iterations needed to be “close enough” to the stationary distribution. Clearly, this is an important value that we would ideally always want to know if we have to sample from a Markov chain, as this will directly tell us how many iterations we need to wait in order for the samples to be drawn from a distribution “sufficiently close” to the stationary distribution. Unfortunately, this value is usually impossible to calculate; to do so, we would need to actually know the distribution $p_{q}^{\left(T\right)}\left(\cdot\right)$ (which is usually very difficult to analytically find), an analytical form for the stationary distribution $\pi\left(\cdot\right)$ (with MCMC methods we usually don't have this) and a measure for the distance between distributions that will be easy to work with (which is non-trivial). Instead of finding the actual mixing time, heuristic approaches are usually used, where the evolution of some statistics from the sampled data are observed throughout the chain, after which a stopping point is chosen by hand. 

Another term that is connected to the mixing time, that is frequently used when talking about MCMC samplers, is the burn-in period. Intuitively, the burn-in period is defined as the number of iterations that have to be “thrown away” at the start of the sampling process in order to be sure the rest of the samples are drawn from the stationary distribution.

{% details Two coins %}
Consider the following distribution:

$$
\begin{equation}
p\left(X,Y\right)=\begin{cases}
X=Y & \frac{1-\epsilon}{2}\\
X\neq Y & \frac{\epsilon}{2}
\end{cases} 
\end{equation}
$$
with $\epsilon>0$ and $X,Y\in\left\{ 0,1\right\}$ . The Gibbs sampling algorithm for this distribution is very simple:

$$
\begin{equation}
X\vert\ Y\sim\begin{cases}
Y & \text{w.p. }1-\epsilon\\
1-Y & \text{w.p. }\epsilon
\end{cases}
\end{equation}
$$

and the probability for $Y\vert\ X$ is the same. 

Even this very simple example already depicts a limitation (that we have to be aware of) inherent to Gibbs sampling. Notice that if we start at either $\left(0,0\right)$ or $\left(1,1\right)$, then the probability of moving to $\left(1,0\right)$ or $\left(0,1\right)$ (the “transition” states) is $\frac{\epsilon}{2}$ at every step. This is exactly the description of a geometric distribution, in which case the expected number of steps to move to either of these “transition” states is:

$$
\mathbb{E}\left[\left(0,0\right)\rightarrow\left\{ \left(0,1\right),\left(1,0\right)\right\} \right]=\frac{1}{\epsilon}
$$

For very small $\epsilon$s this can be a very big number. This means that after initialization, where we will land in either $\left(0,0\right)$ or $\left(1,1\right)$ we will have to wait a very long time until the Gibbs sampler moves from this state. For practical purposes, this means that we have to wait a very long while until our generated samples simulate samples drawn directly from $p\left(X,Y\right)$. 

{% enddetails %}

### Visualizing Mixing Time

The definitions above might make sense mathematically, but I find that hearing about this for the first it's pretty hard to wrap your head around what it actually means. So, let's try to make the definition of mixing times more visceral.

Let's, again, consider a bivariate Gaussian distribution. We want to sample from this distribution using Gibbs sampling, and let's decide that $x^{(0)}=y^{(0)}=0$ - that is, samples are initialized at the origin, whereas the mean of the Gaussian is not at the origin. That is, the initial distribution is $q(x,y)=\delta(x)\delta(y)$. We can visualize the distribution $p^{(t)}_q(x,y)$ in 2D by initializing a lot of points, hundreds of points, and then looking at the form of the distribution that is the result of using the Gibbs sampling algorithm after $t$ iterations. 

When $x$ and $y$ are not very correlated, only a few iterations are needed for the estimated distribution to match the true joint distribution:

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/8_corr.gif"  
alt="Gibbs sampling from a bivariate Gaussian distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 3: Estimated distribution as a function of iterations of the Gibbs sampling algorithm. The plot on the left shows the ground-truth (or target) distribution, overlayed with dots representing the positions of points according to the Gibbs sampling algorithm at each time point. On the right, 500 of these points acquired through the Gibbs sampling algorithm at the same time point are aggregated and their distribution is approximated. Note how after $\sim$25 iterations, the Gibbs sampling already results in a distribution that is very similar to the target distribution.
</div>

In this scenario, where the distribution is "well-behaved", only a few iterations are needed for the Markov chain to mix. But, what happens when $x$ and $y$ are more correlated?

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/99_corr.gif"  
alt="Gibbs sampling from a bivariate Gaussian distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 4: The same as Figure 3, only when the bivariate Gaussian exhibits much higher correlation between the different coordinates. Whereas the previous example required $\sim$25 iterations to approximate the true distribution, here more than 150 iterations are needed for the Markov chain to mix.
</div>

And this can get much worse if there are directions where particles can hardly move at all:

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/999_corr.gif"  
alt="Gibbs sampling from a bivariate Gaussian distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 5: In this extreme case, even 1500 iterations are not enough for the distribution to mix. This situation is a better indication for what happens when sampling from a very high-dimensional distribution that is almost completely determined on a low dimension, which is often the case.
</div>

Finally, another sort of failure mode is when it is very difficult to move from one portion of the distribution to another:

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_15/diag.gif"  
alt="Gibbs sampling from a diagonal Gaussian mixture distribution"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 6: Very far, disconnected "centers" can often be the cause for very slow mixing in Gibbs sampling. Here, points only very very rarely "jump" from the top left center to the bottom right one, as the two centers are disconnected in both $x$ and $y$ axes. 
</div>

There are many ways to try and get around these issues, including more diverse initial conditions, changing the coordinate system before starting the Gibbs sampling procedure, and so on. But, there is no fool-proof way to get around these problems in complex scenarios, which is why it is important to be aware of these issues, get a good intuition for them, and understand why they happen.

---
# Back to  Robust Regression

So, now we have all the pieces we need for Gibbs sampling in the robust regression problem from the start of this post. As a reminder, given a dataset $$\mathcal{D}=\left\{(x_i,y_i)\right\}_{i=1}^N$$, where a fraction $0<\pi_\text{in}<1$ of the points are expected to be inliers, and a set of basis functions $h(x)$, robust regression models the problem as:

$$
\begin{equation}
y_i=\begin{cases}\theta^T h(x_i)+\epsilon_i&\text{w.p.}\ \pi_\text{in}\\ M\epsilon_i&\text{w.p.}\ 1-\pi_\text{in}\end{cases}
\end{equation}
$$

where $\epsilon_i\sim\mathcal{N}(0,\sigma^2)$ and $M\gg 1$. Solving this problem, we would like two outputs:
1. The regression parameters $\theta$
2. Which data points are outliers, represented by the variables $z_1,\cdots,z_N$ where $z_i=0$ means that point $i$ is an inlier and $z_i=1$ means it is an outlier
To get both of these outputs, we will use Gibbs sampling on the joint distribution $p(\theta,z_1,\cdots,z_N\vert\ \mathcal{D})$.

## Conditional Distributions
To determine the steps of the Gibbs sampling procedure, we need to write down each of the conditional distributions. 

Let's start by $\theta$ given all of the $z_i$s:
$$
\begin{equation}
p\left(\theta\vert\ z_1,\cdots,z_N,\mathcal{D}\right)\propto p\left(\mathcal{D}\vert\ \theta,z_1,\cdots,z_N\right)\cdot p\left(\theta\right)
\end{equation}
$$
Notice that if we know whether each point is an inlier or outlier (that is, we know the $z_i$s) and we know $\theta$, then the data points don't depend on each other. In other words:
$$
\begin{align}
p\left(\mathcal{D}\vert\ \theta,z_1,\cdots,z_N\right)&=\prod_{i=1}^N p(y_i\vert\ \theta,z_i)\\
&=\left[\prod_{j:\ z_j=0}\mathcal{N}\left(y_j\vert\ \theta^Th(x_j),\ \sigma^2\right)\right]\left[\prod_{m:\ z_m=1}\mathcal{N}\left(y_m\vert\ 0,\ M\sigma^2\right)\right]
\end{align}
$$

Now, all the terms in this likelihood that don't depend on $\theta$ won't factor into the calculation of the posterior itself, meaning:
$$
\begin{equation}
p\left(\theta\vert\ z_1,\cdots,z_N,\mathcal{D}\right)\propto  p\left(\theta\right)\cdot\prod_{j:\ z_j=0}\mathcal{N}\left(y_j\vert\ \theta^Th(x_j),\ \sigma^2\right)=p(\theta\vert\ \mathcal{D}_\text{inliers})
\end{equation}
$$
which is just the regular linear regression posterior, when ignoring outliers.

Now, we need the distribution of $z_i$ given all variables. Conditioned on the parameters $\theta$, the probability that the $i$-th data point is an inlier is independent of the probability that the $j$-th data point is an inlier, meaning:
$$
\begin{align}
p(z_i=0\vert\ \theta,z_{-i},\mathcal{D})&=p\left(z_i=0\vert\ \theta,y_i\right)\\
&=\frac{p\left(y_i\vert\ z_i=0,\theta\right)p(z_i=0)}{p\left(y_i\vert\ z_i=0,\theta\right)p(z_i=0)+p\left(y_i\vert\ z_i=1,\theta\right)p(z_i=1)}\\
&=\frac{\pi_\text{in}\cdot\mathcal{N}\left(y_i\vert\ \theta^T h(x_i),\ \sigma^2\right)}{\pi_\text{in}\cdot\mathcal{N}\left(y_i\vert\ \theta^T h(x_i),\ \sigma^2\right)+(1-\pi_\text{in})\cdot\mathcal{N}\left(y_i\vert\ 0,\ M\sigma^2\right)}
\end{align}
$$
## Robust Regression with Gibbs Sampling

We now have all the steps we need in order to properly define the Gibbs sampling procedure for robust regression. The full algorithm is given by:

1. for $i=1,\cdots,N$:    $\qquad z_i^{(0)}\sim\text{Bernoulli}(1-\pi_\text{in})$
2. for $t=1,\cdots,T$:
	1. $\mathcal{D}_\text{inliers}=\\{ (x_i,y_i)\vert\ z_i=0 \\}$
	2. $\theta_{(t)}\sim p(\theta\vert\ \mathcal{D}_\text{inliers})$
	3. for $j=1,\cdots, N$:
		1. $p_\text{out}=1-\frac{\pi_\text{in}\cdot\mathcal{N}\left(y_i\vert\ \theta_{(t)}^T h(x_i),\ \sigma^2\right)}{\pi_\text{in}\cdot\mathcal{N}\left(y_i\vert\ \theta_{(t)}^T h(x_i),\ \sigma^2\right)+(1-\pi_\text{in})\cdot\mathcal{N}\left(y_i\vert\ 0,\ M\sigma^2\right)}$
		2. $z^{(t)}\_j\sim\text{Bernoulli}\left(p\_\text{out}\right)$
3. return $\theta_{(T)}$ as the regression parameters, and $z_1^{(T)},\cdots,z_N^{(T)}$ as guesses for inlier/outliers

This seems complicated, but is actually a surprisingly simple algorithm to implement in practice. The only problem is the choice of hyperparameters $\pi_\text{in},M$ and $\sigma$. However, much like $\theta$, it is possible to add prior distributions to these parameters and actually sample them during the Gibbs sampling algorithm as well, making the choice of hyperparameters slightly more flexible.

---
# Discussion

Markov chain Monte Carlo (MCMC) algorithms are some of the most popular sampling methods when confronted with complex distributions. Of these, the most straightforward algorithm is the Gibbs sampling algorithm which we considered in this post. While conceptually straightforward, Gibbs sampling is sometimes very difficult to implement in practice and has many limitations. There exist, of course, numerous extensions to the standard Gibbs sampling algorithm, but I won't really get into them in these posts. 

Instead, in the next post we will consider a much more popular MCMC approach which removes the need for analytically describing the conditional distributions, and instead relies on access to gradients of the distribution, making it incredibly attractive for sampling from distributions described by neural networks.

---
<span style="float:left"><a href="https://friedmanroy.github.io/BML/14_sampling/">← Benefits of sampling</a></span><span style="float:right"><a href=""> →</a></span>
<br>