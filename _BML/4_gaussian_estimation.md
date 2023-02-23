---
layout: distill
comments: true
title: Estimating the Gaussian Distribution
description: The math of estimating the parameters of a Gaussian using MLE as well as Bayesian inference, with some intuition regarding the effects of sample size and prior selection.

authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University

toc:
  - name: ML Estimates
  - name: 1D Bayesian Inference
  - name: Multivariate Gaussian
  - name: Choices of Priors
---

<span style='float:left'><a href="https://friedmanroy.github.io/BML/3_gaussians/">← The Gaussian Distribution</a></span><span style='float:right'><a href="https://friedmanroy.github.io/BML/5_linear_regression/">Linear Regression →</a></span>
<br>

While Bayesian statistics is our main interest in this thread of posts, many times it will prove easier to first go over the frequentist version as it is less mathematically involved. Only after we understand the ML solution, we will move on to the Bayesian treatment of the same, in the process revealing how they are related to each other.

The parameters of a Gaussian distribution are $\mu$ and $\Sigma$, so $\theta=\left\{ \mu,\Sigma\right\}$ . In the frequentist case we will estimate both, however the Bayesian treatment of $\Sigma$ is a bit more complex and doesn't teach much, so we will ignore it for now. 

<br>
# ML Estimates

The log-likelihood of a data set $\mathcal{D}=\left\{ x_{i}\right\} _{i=1}^{N}$ sampled from a Gaussian distribution is:

$$
\begin{equation}
\begin{split}\log p\left(\mathcal{D}|\mu,\Sigma\right) & =\sum_{i}\log\mathcal{N}\left(x_{i}\,|\,\mu,\Sigma\right)\\
 & =\sum_{i}\log\left[\frac{1}{\sqrt{\left(2\pi\right)^{d}\left|\Sigma\right|}}\exp\left\{ -\frac{1}{2}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)\right\} \right]\\
 & =\sum_{i}\left[-\frac{d}{2}\log2\pi-\frac{1}{2}\log\left|\Sigma\right|-\frac{1}{2}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)\right]\\
 & =-\frac{Nd}{2}\log2\pi-\frac{N}{2}\log\left|\Sigma\right|-\frac{1}{2}\sum_{i}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)
\end{split}
\end{equation}
$$

Before we begin the process of finding the ML estimators<d-footnote>Bishop 2.3.4; Murphy 4.1.3.</d-footnote> for $\mu$ and $\Sigma$ , let's see another way of writing the log-likelihood. Notice that for any scalar $a$ , we can write:
$$
\begin{equation}
a=\text{trace}\left[a\right]
\end{equation}
$$
Since this is true for any scalar, we can apply this to the inner product of 2 vectors $x^{T}y$ (which is just a number), as well:

$$
\begin{equation}
x^{T}y=\text{trace}\left[x^{T}y\right]=\text{trace}\left[yx^{T}\right]
\end{equation}
$$

Now, recall that the Mahalanobis distance $\left(x-\mu\right)^{T}\Sigma^{-1}\left(x-\mu\right)$ is also a scalar, so we can use the above identity to rewrite it as:

$$
\begin{align}
\ell\left(\mathcal{D}|\mu,\Sigma\right) & =-\frac{Nd}{2}\log2\pi-\frac{N}{2}\log\left|\Sigma\right|-\frac{1}{2}\sum_{i}\text{trace}\left[\Sigma^{-1}\left(x_{i}-\mu\right)\left(x_{i}-\mu\right)^{T}\right]\nonumber \\
 & =-\frac{Nd}{2}\log2\pi-\frac{N}{2}\log\left|\Sigma\right|-\frac{1}{2}\text{trace}\left[\Sigma^{-1}\sum_{i}\left(x_{i}-\mu\right)\left(x_{i}-\mu\right)^{T}\right]
\end{align}
$$

Finally, if we define $S\stackrel{\Delta}{=}\frac{1}{N}\sum_{i}\left(x_{i}-\mu\right)\left(x_{i}-\mu\right)^{T}$ (which is almost the empirical covariance), we get a shorter form for the log-likelihood (which is sometimes used in the literature):

$$
\begin{equation}
\ell\left(\mathcal{D}|\mu,\Sigma\right)=-\frac{N}{2}\left(d\log2\pi+\log\left|\Sigma\right|+\text{trace}\left[\Sigma^{-1}S\right]\right)
\end{equation}
$$

## MLE for $\mu$ <a name="mu-MLE"></a>

We begin by finding the mean that maximizes the log-likelihood, by differentiating the log-likelihood:

$$
\begin{align}
\frac{\partial\ell}{\partial\mu} & =-\frac{1}{2}\sum_{i}\frac{\partial}{\partial\mu}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)\\
 & =-\frac{1}{2}\sum_{i}2\Sigma^{-1}\left(x_{i}-\mu\right)\\
 & =-\Sigma^{-1}\sum_{i}\left(x_{i}-\mu\right)\stackrel{!}{=}0
\end{align}
$$

By equating to 0 we can find the maxima:

$$
\begin{equation}
\hat{\mu}_{\text{ML}}=\frac{1}{N}\sum_{i}x_{i}
\end{equation}
$$

Here we write $\hat{\mu}\_{\text{ML}}$ to show that it is the _maximum likelihood estimator_ for the data set. Notice that, unsurprisingly, the ML estimator for the mean of the Gaussian
is the _empirical mean_ of the data.


## MLE for $\Sigma$

Using the following definition of the derivatives (which are a bit harder to get directly on your own):

$$
\begin{equation}
\frac{\partial}{\partial\Sigma}\log\left|\Sigma\right|=\frac{1}{\left|\Sigma\right|}\frac{\partial}{\partial\Sigma}\left|\Sigma\right|=\frac{1}{\left|\Sigma\right|}\left|\Sigma\right|\Sigma^{-1}=\Sigma^{-1}
\end{equation}
$$

and:

$$
\begin{equation}
\frac{\partial}{\partial\Sigma}\text{trace}\left[\Sigma^{-1}S\right]=-\Sigma^{-1}S\Sigma^{-1}
\end{equation}
$$

we can find the MLE for $\Sigma$. The full derivative of the log-likelihood by $\Sigma$ is:

$$
\begin{align}
\frac{\partial\ell}{\partial\Sigma} & =-\left(\Sigma^{-1}-\Sigma^{-1}S\Sigma^{-1}\right)\stackrel{!}{=}0\nonumber \\
\Rightarrow & \Sigma^{-1}=\Sigma^{-1}S\Sigma^{-1}\nonumber \\
\Rightarrow & I=\Sigma^{-1}S\nonumber \\
\Rightarrow & \Sigma=S\nonumber \\
\Rightarrow & \hat{\Sigma}_{\text{ML}}=\frac{1}{N}\sum_{i}\left(x_{i}-\mu\right)\left(x_{i}-\mu\right)^{T}
\end{align}
$$

Because $\hat{\mu}\_{\text{ML}}$ is not dependent on $\hat{\Sigma}\_{\text{ML}}$, we can first find the MLE for $\mu$ and then for $\Sigma$ , so that:

$$
\begin{equation}
\hat{\Sigma}_{\text{ML}}=\frac{1}{N}\sum_{i}\left(x_{i}-\hat{\mu}_{\text{ML}}\right)\left(x_{i}-\hat{\mu}_{\text{ML}}\right)^{T}
\end{equation}
$$

---

Putting the two equations together, the MLE for a Gaussian distribution are:
$$
\begin{equation}
\begin{split}\hat{\mu}_{\text{ML}} & =\frac{1}{N}\sum_{i}x_{i}\\
\hat{\Sigma}_{\text{ML}} & =\frac{1}{N}\sum_{i}\left(x_{i}-\hat{\mu}_{\text{ML}}\right)\left(x_{i}-\hat{\mu}_{\text{ML}}\right)^{T}
\end{split}
\end{equation} 
$$


<br>
# 1D Bayesian Inference

Recall that in the Bayesian treatment, we assume that the parameters are distributed in some manner. We begin by considering the 1D case for Gaussian distributions<d-footnote>See Bishop 2.3.6 for more details.</d-footnote>:

$$
\begin{equation}
p\left(x\right)=\frac{1}{Z}\exp\left[-\frac{\left(x-\mu\right)^{2}}{2\sigma^{2}}\right]
\end{equation}
$$

For now, we will assume that we know the variance $\sigma^{2}$ . We will assume a Gaussian prior over $\mu$ (if we want we can assume different priors as well, but let's stick with Gaussian priors for now):

$$
\begin{equation}
p\left(\mu\right)=\mathcal{N}\left(\mu_{0},\sigma_{0}^{2}\right)
\end{equation}
$$

Given a data set $\mathcal{D}=\left\{ x_{i}\right\} _{i=1}^{N}$ , the likelihood is:

$$
\begin{align}
p\left(\mathcal{D}|\mu\right) & =\prod_{i=1}^{N}p\left(x_{i}|\mu\right)=\prod_{i=1}^{N}\mathcal{N}\left(x_{i}\,|\,\mu,\sigma^{2}\right)\\
 & \propto\exp\left[-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]
\end{align}
$$

The posterior probability for $\mu$ will then be:

$$
\begin{align}\label{eq:post}
p\left(\mu|\mathcal{D}\right) & =\frac{p\left(\mathcal{D}|\mu\right)p\left(\mu\right)}{p\left(\mathcal{D}\right)}\\
 & \propto p\left(\mathcal{D}|\mu\right)p\left(\mu\right)
\end{align}
$$

Recall that the term $p\left(\mathcal{D}\right)$ is constant and only serves as a normalization, so for now we can ignore it. 

Let's look at the product in equation \eqref{eq:post} more closely:

$$
\begin{align}
p\left(\mathcal{D}|\mu\right)p\left(\mu\right) & \propto\exp\left[-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]\exp\left[-\frac{\left(\mu-\mu_{0}\right)^{2}}{2\sigma_{0}^{2}}\right]\\
 & =\exp\left[-\frac{1}{2}\left(\frac{1}{\sigma^{2}}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}+\frac{1}{\sigma_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}\right)\right]
\end{align}
$$

Notice that the term in the exponent is _still quadratic in_ $\mu$ . This means, of course, that this whole term is still a Gaussian distribution. Let's use the derivative trick [from the previous post](https://friedmanroy.github.io/BML/3_gaussians/) in order to find the distribution of $\mu$ exactly. Define:

$$
\begin{equation}
\Delta\stackrel{\Delta}{=}\frac{1}{2}\left(\frac{1}{\sigma^{2}}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}+\frac{1}{\sigma_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}\right)
\end{equation}
$$

Recall, we can now differentiate $\Delta$ with respect to $\mu$ in order to find the mean and covariance of the posterior distribution:

$$
\begin{align}
\frac{\partial\Delta}{\partial\mu} & =\frac{1}{2}\left(\frac{1}{\sigma^{2}}\sum_{i}\frac{\partial}{\partial\mu}\left(x_{i}-\mu\right)^{2}+\frac{1}{\sigma_{0}^{2}}\frac{\partial}{\partial\mu}\left(\mu-\mu_{0}\right)^{2}\right)\nonumber \\
 & =\frac{1}{\sigma^{2}}\sum_{i}\left(\mu-x_{i}\right)+\frac{1}{\sigma_{0}^{2}}\left(\mu-\mu_{0}\right)\nonumber \\
 & =\frac{1}{\sigma^{2}}\left(N\mu-\sum_{i}x_{i}\right)+\frac{1}{\sigma_{0}^{2}}\left(\mu-\mu_{0}\right)\nonumber \\
 & =\frac{N}{\sigma^{2}}\left(\mu-\mu_{ML}\right)+\frac{1}{\sigma_{0}^{2}}\left(\mu-\mu_{0}\right)\nonumber \\
 & =\left(\frac{N}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\right)\left(\mu-\left(\frac{N}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\right)^{-1}\left(\frac{N}{\sigma^{2}}\mu_{\text{ML}}+\frac{1}{\sigma_{0}^{2}}\mu_{0}\right)\right)
\end{align}
$$

where $\mu_{\text{ML}}=\frac{1}{N}\sum_{i}x_{i}$ is the ML estimate for $\mu$ , as we showed in [section for the MLE of the mean](#mu-MLE) . 


Defining: 
$$
\begin{equation}
\frac{N}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\stackrel{\Delta}{=}\frac{1}{\sigma_{N}^{2}}
\end{equation}
$$
the posterior of $\mu$ is equal to:

$$
\begin{equation}
p\left(\mu|\mathcal{D}\right)=\mathcal{N}\left(\mu\,|\,\sigma_{N}^{2}\left(\frac{N}{\sigma^{2}}\mu_{\text{ML}}+\frac{1}{\sigma_{0}^{2}}\mu_{0}\right),\,\sigma_{N}^{2}\right)
\end{equation}
$$

where $\sigma_{N}^{2}=\left(\frac{N}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\right)^{-1}=\left(\frac{N\sigma_{0}^{2}+\sigma^{2}}{\sigma_{0}^{2}\sigma^{2}}\right)^{-1}=\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}$ and $\mu\_{\text{ML}}=\frac{1}{N}\sum_{i}x_{i}$ . If we write all of this explicitly, we will get:

$$
\begin{equation}
p\left(\mu|\mathcal{D}\right)=\mathcal{N}\left(\mu\,|\,\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\left(\frac{N}{\sigma^{2}}\mu_{\text{ML}}+\frac{1}{\sigma_{0}^{2}}\mu_{0}\right),\,\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\right)
\end{equation}
$$


## Effects of Sample Size

It may be a good idea to get some intuition for the posterior we found. Let's look at the slightly simpler case of $\mu_{0}=0$ (but the analysis that follows is true for any $\mu_{0}$ ). In this case, the posterior is:

$$
\begin{equation}
\mathcal{N}\left(\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\cdot\frac{N}{\sigma^{2}}\mu_{\text{ML}},\,\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\right)
\end{equation}
$$

Let's see what happens when $N=0$ . If we don't have any data, we should probably always fall back to the only thing we know; our prior. At $N=0$ , we have:

$$
\begin{equation}
N=0\qquad\begin{array}{c}
\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\cdot\frac{N}{\sigma^{2}}\mu_{\text{ML}}=0=\mu_{0}\\
\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}=\frac{\sigma_{0}^{2}\sigma^{2}}{\sigma^{2}}=\sigma_{0}^{2}
\end{array}
\end{equation}
$$

so the posterior (naturally) falls back to the prior. If we look at the other extreme, $N\rightarrow\infty$ , then there should be no ambiguity over the value of $\mu$ whatsoever:

$$
\begin{equation}
N\rightarrow\infty\qquad\begin{array}{c}
\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\cdot\frac{N}{\sigma^{2}}\mu_{\text{ML}}\rightarrow\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}}\cdot\frac{N}{\sigma^{2}}\mu_{\text{ML}}=\mu_{\text{ML}}\\
\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}+\sigma^{2}}\rightarrow\frac{\sigma_{0}^{2}\sigma^{2}}{N\sigma_{0}^{2}}=0
\end{array}
\end{equation}
$$

If $N$ is somewhere in between, then the term for the mean is (as we saw before):
$$
\begin{equation}\label{eq:mu-N}
\mu_{N}\stackrel{\Delta}{=}\left(\frac{N}{\sigma^{2}}+\frac{1}{\sigma_{0}^{2}}\right)^{-1}\left(\frac{N}{\sigma^{2}}\mu_{\text{ML}}+\frac{1}{\sigma_{0}^{2}}\mu_{0}\right)
\end{equation}
$$
This is a _weighted mean_ of the two values $\mu\_{\text{ML}}$ and $\mu_{0}$ (you can find a demo for this behavior [here](https://www.desmos.com/calculator/dgrvldq2ok)). We can look at the number of samples needed in order for $\mu\_{N}$ to be _exactly_ between the ML estimate and the prior by giving equal weight to both terms:

$$
\begin{equation}
\frac{N}{\sigma^{2}}\stackrel{!}{=}\frac{1}{\sigma_{0}^{2}}\Rightarrow\hat{N}=\frac{\sigma^{2}}{\sigma_{0}^{2}}
\end{equation}
$$

So, when the variance of the prior is very small, which is like saying "we are very sure that $\mu$ is close to $\mu\_{0}$ ", then a lot of samples are needed in order to move $\mu\_{N}$ away from the prior $\mu\_{0}$ . If, on the other hand, the variance of the prior is very large, which may mean we are very unsure that $\mu\_{0}$ is correct, then few points are needed in order to move the mean from the prior mean. Finally, if the sample variance ( $\sigma^{2}$ ) is very large, then we need to get a lot of data to be sure that the MLE is correct, while if it is very small, then we need very few points in order to be sure of the MLE. 

Because the posterior is so dependent on the number of samples, it is sometimes written (like in equation \eqref{eq:mu-N}) as:

$$
\begin{equation}
p\left(\mu|\mathcal{D}\right)=\mathcal{N}\left(\mu\,|\,\mu_{N},\sigma_{N}^{2}\right)
\end{equation}
$$

with the intention behind this notation being "this is the posterior
mean after having sampled $N$ points".


<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_3/1D_sample_size.png"  
alt="Effects of sample size on the posterior of the mean of a 1D Gaussian"  
style="display: inline-block; margin: 0 auto; ">
</p>
<div class="caption">
    Figure 1: Posteriors for the mean after different amount of data are seen. When $N=0$ , the posterior is equal to the prior. As more points are observed, the posterior is pulled towards the true value that generated the points. Brighter posteriors are those with more observed points.
</div>


## MAP and MMSE Estimates for $\mu$

If we want to find the MAP estimate for $\mu$ under the prior above, we need to find:

$$
\begin{equation}
\hat{\mu}_{\text{MAP}}=\arg\max_{\mu}p\left(\mu|\mathcal{D}\right)=\arg\max_{\mu}\mathcal{N}\left(\mu\,|\,\mu_{N},\sigma_{N}^{2}\right)
\end{equation}
$$

Of course, the Gaussian distribution only has one maxima, which is the mean of the distribution. So the MAP estimate of $\mu$ is simply the mean:

$$
\begin{equation}
\hat{\mu}_{\text{MAP}}=\mathbb{E}\left[p\left(\mu|\mathcal{D}\right)\right]=\mu_{N}
\end{equation}
$$

where $\mu\_{N}$ is given explicitly in equation \eqref{eq:mu-N}. Notice that (in this case) this is also the MMSE estimate:
$$
\begin{equation}
\hat{\mu}_{MAP}=\mathbb{E}\left[p\left(\mu|\mathcal{D}\right)\right]=\hat{\mu}_{MMSE}
\end{equation}
$$
The fact that the MAP and MMSE estimates are the same is unique to the Gaussian distribution, in general they will differ quite a bit!

<br>
# Multivariate Gaussian

Now that we understood the basic premise of the Bayesian inference for $\mu$ in 1D, we can start all over again for the multivariate case. We assume, again, that:
$$
\begin{equation}
x\sim\mathcal{N}\left(\mu,\Sigma\right)
\end{equation}
$$
where the covariance matrix $\Sigma$ is known. We also assume a prior over $\mu$ of the form:
$$
\begin{equation}
\mu\sim\mathcal{N}\left(\mu_{0},\Sigma_{0}\right)
\end{equation}
$$
The likelihood for a data set $\mathcal{D}$ is:
$$
\begin{equation}
p\left(\mathcal{D}|\mu\right)\propto\exp\left[-\frac{1}{2}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)\right]
\end{equation}
$$
and the posterior is:
$$
\begin{equation}
p\left(\mu|\mathcal{D}\right)\propto\exp\left[-\frac{1}{2}\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{T}\Sigma^{-1}\left(x_{i}-\mu\right)\right]\exp\left[-\frac{1}{2}\left(\mu-\mu_{0}\right)^{T}\Sigma_{0}^{-1}\left(\mu-\mu_{0}\right)\right]
\end{equation}
$$

Essentially nothing has changed from before; the term in the exponent is still quadratic in $\mu$, so we can employ our tricks once again:

$$
\begin{align*}
\frac{\partial}{\partial\mu}\Delta & =\Sigma^{-1}\sum_{i}\left(\mu-x_{i}\right)+\Sigma_{0}^{-1}\left(\mu-\mu_{0}\right)\\
 & =N\Sigma^{-1}\left(\mu-\mu_{\text{ML}}\right)+\Sigma_{0}^{-1}\left(\mu-\mu_{0}\right)\\
 & =\left(N\Sigma^{-1}+\Sigma_{0}^{-1}\right)\left[\mu-\left(N\Sigma^{-1}+\Sigma_{0}^{-1}\right)^{-1}\left(N\Sigma^{-1}\mu_{\text{ML}}+\Sigma_{0}^{-1}\mu_{0}\right)\right]
\end{align*}
$$

where we used the same definition as before for $\mu\_{\text{ML}}$ (the ML estimate for $\mu$ ). The full posterior is given by:
$$
\begin{equation}
\mu|\mathcal{D}\sim\mathcal{N}\left(\mu_{N},\Sigma_{N}\right)
\end{equation}
$$
where:
$$
\begin{align}
\Sigma_{N} & =\left[N\Sigma^{-1}+\Sigma_{0}^{-1}\right]^{-1}\\
\mu_{N} & =\Sigma_{N}\left[N\Sigma^{-1}\mu_{\text{ML}}+\Sigma_{0}^{-1}\mu_{0}\right]
\end{align}
$$ 

The result is consistent with what we saw in 1D. The main difference here is that now we need to invert the matrix $N\Sigma^{-1}+\Sigma_{0}^{-1}$ in order to find $\Sigma\_{N}$ and $\mu\_{N}$ . The MAP/MMSE estimates for the multivariate $\mu$ are again the mean of the posterior (since this is a Gaussian distribution as well):
$$
\begin{equation}
\hat{\mu}_{\text{MAP}}=\left[N\Sigma^{-1}+\Sigma_{0}^{-1}\right]^{-1}\left[N\Sigma^{-1}\mu_{\text{ML}}+\Sigma_{0}^{-1}\mu_{0}\right]=\hat{\mu}_{MMSE}
\end{equation}
$$



<br>
# Choices of Priors

Bayesian machine learning is often described in terms of "known priors". However, many times we don't actually have an explicit prior we can choose, which is the main criticism against the Bayesian approach. The problem is that when there isn't a good prior, researchers amount to choosing arbitrary distributions for their priors. 



<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_3/high_dim_fitting.png"  
alt="Estimating the mean of an MVN in high dimensions under different  priors"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 2: estimating the mean of a 30-dimensional Gaussian under different priors; to generate the data $\mu=0$ was used, while the near prior mean was $\mu_{0}=1$ and the far prior mean was $\mu_{0}=10$. The shaded areas are the areas of the posterior with total probability of 95%. This simple example shows that the prior can positively or negatively affect the performance of the estimation.
</div>


The figure above illustrates what happens when Gaussian priors of different kinds are chosen. When the variance of the prior is low, i.e. $\Sigma\_{0}=I\sigma\_{0}^{2}$ with small $\sigma$ (left column), then many samples are needed to change the posterior distribution. When the prior mean is well calibrated to the generating distribution, this translates to a better MMSE than the ML estimate with few samples but doesn't get better when more samples are introduced. However, when the prior mean is far from the generating distribution, then the estimate will always be quite bad. The other end of the spectrum is when $\sigma$ is large (right column), in which case it doesn't really matter what the prior mean is since the posterior mean is more or less equal to the ML estimate.

The more interesting case is when $\mu\_{0}$ is well calibrated and $\Sigma\_{0}$ is moderate (middle column, top). In this setting, the MMSE gives a much better estimate than the MLE, _especially_ in low sample-size settings - in this case, more than an order of magnitude. However, when the prior is bad, the MMSE estimate will always be worse than the MLE (middle column, bottom).

<br>

<span style='float:left'><a href="https://friedmanroy.github.io/BML/3_gaussians/">← The Gaussian Distribution</a></span><span style='float:right'><a href="https://friedmanroy.github.io/BML/5_linear_regression/">Linear Regression →</a></span>