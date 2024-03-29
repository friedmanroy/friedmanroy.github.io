---
layout: distill
comments: false
title: Decision Theory and Bayes-Optimal Estimators
description: An overview of Bayesian decision theory. As part of this post, we will look into estimation as well as proofs for the MLE, MMSE and MAP estimators.
date: 2022-10-28
authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University
toc:
  - name: Frequentist Evaluation
    subsections:
      - name: Maximum Likelihood Estimation
  - name: Bayesian Evaluation
    subsections:
      - name: Minimum MSE Estimator
      - name: Other Bayes-Optimal Estimators
      - name: The 0-1 Loss
  - name: Discussion
---

<span style='float:left'><a href="https://friedmanroy.github.io/BML/1_philosophy/">← The Bayesian Philosophy</a></span><span style='float:right'><a href="https://friedmanroy.github.io/BML/3_gaussians/">The Gaussian Distribution →</a></span>
<br>
<br>

> In the [previous post](https://friedmanroy.github.io/BML/1_philosophy/) we saw what the frequentist and Bayesian philosophies are, how they are different, and also a bit of how they are similar. In this post, we will take a look at the specific task of estimation and methods to analyze the optimality of estimators in both regimes.

Suppose we see some data $\mathcal{D}$ which we assume was generated according to some parameter $\theta$ . The task of estimation is to estimate $\theta$ given the observed data $\mathcal{D}$ . More precisely, in estimation we want to define a function which returns a guess for what $\theta$ could have been given the observed data. Many times, this algorithm is concisely notated as $\hat{\theta}(\mathcal{D})$ , where the   $\hat{}$   is to remember that it is an estimate.

---

#### Example: Coin Toss

The simplest of all examples is the parameter of a coin. We observe the data $\mathcal{D}=\\{1,1,0,\cdots, 0, 0, 1\\}$  , where a 1 is heads and a 0 is tails. The parameterization of this problem is:

$$
\begin{equation}
p(1\vert \theta)=\theta \Leftrightarrow p(0\vert \theta)=1-\theta 
\end{equation}
$$
where $\theta \in [0,1]$ . That is, the probability for heads is $\theta$ . Given the dataset $\mathcal{D}$ , we want to find the original $\theta$ that generated this data.

---

What follows is a decision-theoretic approach for choosing how to estimate the parameters. This will allow us to give concrete guarantees regarding specific estimators and gives a general framework in order to determine when an estimator is optimal while also elucidating what it means for an estimator to be optimal. 

<br>

# Frequentist Evaluation

The classical approach to this would be to devise a _loss_ $\mathcal{L}(\hat{\theta},\theta^\star;\ \mathcal{D})$  <d-footnote>This notation for the loss function is not the typical notation that is used and is a bit cumbersome, but it is useful as it shows which information we assume that we have when calculating the loss.</d-footnote> which allows us to evaluate the estimator $\hat{\theta}(\mathcal{D})$ against the true parameter value $\theta^\star$ . Written this way, the estimator is _an algorithm_ that has as it's input the dataset $\mathcal{D}$ and outputs a guess of the true parameters. The function $\mathcal{L}(\cdot ,\theta^\star;\ \mathcal{D})$ tells us how much "we are losing" by using the estimator $\hat{\theta}(\mathcal{D})$ .

Of course, we aren't only interested in how well our estimator performs on a specific dataset, but instead we want to evaluate the accuracy for any set of points. To do so, we can use the following function:

$$
\begin{equation}
\mathcal{R}(\hat{\theta}\vert \theta^\star)=\mathbb{E}_{\mathcal{D}\sim p(\mathcal{D}\vert \theta^\star)}\left[\mathcal{L}(\hat{\theta},\ \theta^\star;\ \mathcal{D})\right]
\end{equation}
$$

where the expectation with respect to $\mathcal{D}\sim p(\mathcal{D}\vert \theta^\star)$ means the expectation with respect to datasets generated from the ground truth parameter, $\theta^\star$ . The function $\mathcal{R}(\cdot\vert \theta^\star)$ is called the _risk_ of the estimator and it allows us to evaluate how good of an estimator $\hat{\theta}$ is on average; in different words, $\mathcal{R}(\hat{\theta}\vert \theta^\star)$ tells us how much (on average) we are risking by using the estimator $\hat{\theta}$ . 

As a particular example, one of the most common loss functions is the squared error function:

$$
\begin{equation}
\mathcal{L}(\hat{\theta},\ \theta^\star;\ \mathcal{D}) = \vert\vert \theta^\star-\hat{\theta}(\mathcal{D})\vert\vert ^2
\end{equation}
$$

in which case the corresponding risk function is called the _mean squared error_ (MSE), defined as:

$$
\begin{equation}
\text{MSE}\left(\hat{\theta}\vert \theta^\star\right)\stackrel{\Delta}{=}\mathbb{E}_{\mathcal{D}\vert \theta^\star}\left[\vert \vert \theta^\star-\hat\theta(\mathcal{D})\vert \vert ^2\right]
\end{equation}
$$


Notice that this construction allows us to evaluate algorithms under specific values of $\theta^\star$ , but we can't use these in order to find an algorithm that is optimal under every possible parameterization, since they are all dependent on $\theta^\star$ .

--- 

We've made a couple of assumptions so far. First, we assumed that the data is truly generated by the parametric form we chose; that is, the data was really generated using a set of parameters $\theta^\star$. Second, we assumed that _there is one and only one true parameter_ $\theta^\star$ . Given these assumptions, the risk is an adequate evaluator for the algorithm $\hat{\theta}(\mathcal{D})$ .

Furthermore, this gives us a method to find "optimal" estimators. Finding an estimator $\hat{\theta}(\mathcal{D})$ that minimizes a specific risk function will give us an estimator that is optimal with respect to the corresponding loss function. Such an estimator will ensure that the loss is, on average, as small as possible.


## Maximum Likelihood Estimation

One of the most commonly used estimators is the _maximum likelihood estimator_ (MLE). Given data $\mathcal{D}$ and a likelihood function $p(\mathcal{D}\vert \theta)$ , the MLE is given by:

$$
\begin{equation}
\hat{\theta}_{\text{ML}}\stackrel{\Delta}{=}\arg\max_\theta p(\mathcal{D}\vert \theta)=\arg\max_\theta\left\{\log p(\mathcal{D}\vert \theta)\right\}
\end{equation}
$$

Intuitively, using this estimator makes a lot of sense, right? The estimator $\hat{\theta}_{\text{ML}}$ is the one the gives the highest probability to the data, so it seems that it might be close to the true parameters that generated the data in some sense. We will see that this intuition turns out to be correct, assuming that there's one true parameter value $\theta^\star$ . In order to do so, we will define the appropriate loss function and the corresponding risk.

Notice that:

$$
\begin{align}
\arg\max_\theta\left\{\log p(\mathcal{D}\vert \theta)\right\}&=\arg\max_\theta\left\{\frac{\log p(\mathcal{D}\vert \theta)}{\log p(\mathcal{D}\vert \theta^\star)}\right\}\\
&=\arg\min_\theta\left\{-\frac{\log p(\mathcal{D}\vert \theta)}{\log p(\mathcal{D}\vert \theta^\star)}\right\}\\
&\stackrel{\Delta}{=}\arg\min_\theta\mathcal{L}_{\text{ML}}\left(\theta,\theta^\star;\ \mathcal{D}\right)
\end{align}
$$

That is, maximizing the likelihood is equivalent to minimizing the loss $\mathcal{L}_{\text{ML}}\left(\cdot,\cdot;\ \mathcal{D} \right)$ . The ML-risk is the following function:

$$
\begin{equation}\label{eq:KL-risk}
	\mathcal{R}_{\text{ML}}(\theta\vert \theta^\star)=D_{\text{KL}}(\theta^\star\vert \vert \theta)=-\mathbb{E}_{\mathcal{D}\vert \theta^\star}\left[\frac{\log p(\mathcal{D}\vert \theta)}{\log p(\mathcal{D}\vert \theta^\star)}\right]
\end{equation}
$$

As it turns out, the function $D_{\text{KL}}(\theta^\star\vert \vert \theta)$ is a divergence between distributions; a measure of how different the two distributions are from each other. This divergence is called the [_Kullback-Leibler_ (KL-)divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). For two distributions $\mathcal{P}$ and $\mathcal{Q}$ , this divergence has the two following (important) properties:

$$
\begin{align}
D_{\text{KL}}(\mathcal{P}\vert \vert \mathcal{Q})&\ge 0\\
D_{\text{KL}}(\mathcal{P}\vert \vert \mathcal{Q})=0 &\Leftrightarrow \mathcal{P}=\mathcal{Q}
\end{align}
$$

That is, the smaller the KL-divergence, the closer the distributions are to each other and they are equal if and only if the KL-divergence is equal to 0. 

So, minimizing the risk in equation \eqref{eq:KL-risk} is equivalent to finding the parameters $\theta$ under which the likelihood is closest to the likelihood under the true parameters $\theta^\star$ . However, the MLE we saw before doesn't exactly do this - there's the issue of the expectation, which isn't taken into account in the MLE. This leads us to the following theorem. 


> **Theorem**: Let $p(x\vert \theta)$ be some likelihood function over the random variable $x$ such that that for any two random variables $x_1$ and $x_2$ they are independent conditionally on $\theta$ , i.e. $p(x_1,x_2\vert \theta)=p(x_1\vert \theta)p(x_2\vert \theta)$ . 
> 
> Given a dataset $\mathcal{D}=\{x_i\}_{i=1}^N$ , assume that there exists some true parameter $\theta^\star$ such that $\forall i\ \ x_i\sim p(x\vert \theta^\star)$ . Then, the MLE is optimal in terms of the ML-risk in equation \eqref{eq:KL-risk} when $N\rightarrow\infty$ .


<br>
{%details Click here to see the proof%}
#### Proof:

By construction, notice that:
$$
\begin{align}
\arg\max_\theta\left\{\log p(\mathcal{D}\vert \theta)\right\} &=\arg\max_\theta\left\{\frac{1}{N}\sum_{i=1}^N\log p(x_i\vert \theta)\right\}\\
&=\arg\min_\theta\left\{-\frac{1}{N}\sum_{i=1}^N\frac{\log p(x_i\vert \theta)}{\log p(x_i\vert \theta^\star)}\right\}\\
&\stackrel{N\rightarrow\infty}{=}\arg\min_\theta\left\{-\mathbb{E}_{x\vert \theta^\star}\left[\frac{\log p(x_i\vert \theta)}{\log p(x_i\vert \theta^\star)}\right]\right\}\\
&=\arg\min_\theta\mathcal{R}_{\text{ML}}(\theta\vert \theta^\star)
\end{align}
$$
In other words, $\hat{\theta}_{\text{ML}}$ minimizes the KL-divergence to the true distribution (if it exists).
<span style='float:right'> $\square$ </span>

{% enddetails %}
<br>

# Bayesian Evaluation

The assumption that the data is always generated from one specific parameter $\theta^\star$ is quite constricting. A more general assumption is that we have some distribution over which $\theta$'s we assume are more or less likely. This distribution is called the prior, $p(\theta)$ . In this setting, we assume that for each possible dataset a different parameter was chosen, and each of them was sampled from the prior.

In this case, low risk is not enough to guarantee that the algorithm we chose is good, since it takes into account only one parameter value. We want to make sure that the estimator is accurate _across all possible datasets_ and _for all possible choices of_ $\theta$ according to the prior. To do so, we will introduce a new way to evaluate estimators, called the _Bayesian risk_:
$$
\begin{equation}
\mathcal{R}(\hat{\theta})=\mathbb{E}_{\theta}\left[\mathcal{R}\left(\hat{\theta}\vert \theta\right)\right]=\mathbb{E}_{\theta,\mathcal{D}}\left[\mathcal{L}\left(\hat{\theta},\theta;\ \mathcal{D}\right)\right]
\end{equation}
$$
This new risk takes into account all possible combinations of datasets and parameters.



## Minimum MSE Estimator

The corresponding Bayesian risk to the MSE we saw above is the _Bayesian MSE_ (BMSE):

$$
\begin{equation}
\text{BMSE}(\hat{\theta})\stackrel{\Delta}{=}\mathbb{E}_{\theta,\mathcal{D}}\left[\vert \vert \theta-\hat{\theta}(\mathcal{D})\vert \vert ^2\right]
\end{equation}
$$

Using this definition, we can find the estimator that is optimal in the BMSE sense; that is, the estimator that achieves the best MSE for any possible choice of parameter $\theta$ . This estimator is called the _minimum MSE_ (MMSE) estimator.

> **Theorem**: $\hat\theta(\mathcal{D})=\mathbb{E}[\theta\vert \mathcal{D}]$ is the MMSE estimator; that is, the optimal estimator in terms of the BMSE.


<br>
{%details Click here to see the proof%}
#### Proof:

Let's start with the definition of the BMSE:

$$
\begin{align}
\text{BMSE}\left(\hat{\theta}\right) & =\mathbb{E}_{\theta,\mathcal{D}}\left[\vert \vert \theta-\hat{\theta}(\mathcal{D})\vert \vert ^{2}\right]\\
 & =\intop p\left(\theta,\mathcal{D}\right)\vert \vert \theta-\hat{\theta}\left(\mathcal{D}\right)\vert \vert ^{2}d\theta d\mathcal{D}\\
 & =\int p\left(\mathcal{D}\right)\left(\intop p\left(\theta\vert \mathcal{D}\right)\vert \vert \theta-\hat{\theta}\left(\mathcal{D}\right)\vert \vert ^{2}d\theta\right)d\mathcal{D}\\
 & \stackrel{\Delta}{=}\int p\left(\mathcal{D}\right)J\left(\theta\vert \mathcal{D}\right)d\mathcal{D}
\end{align}
$$

Where  $J(\theta\vert \mathcal{D})=\mathbb{E}_{\theta\vert \mathcal{D}}[\vert \vert \theta-\hat{\theta}\left(\mathcal{D}\right)\vert \vert ^{2}]$ . 

Notice that if we find an estimator that minimizes $J(\theta\vert \mathcal{D})$ for every dataset $\mathcal{D}$ , then this estimator will also be the one that minimizes the BMSE. Since $J(\theta\vert \mathcal{D})$ is quadratic with respect to $\hat\theta(\mathcal{D})$, we can differentiate and equate to zero:

$$
\begin{align}
\frac{\partial J}{\partial\hat{\theta}} & =\frac{\partial}{\partial\hat{\theta}}\intop p\left(\theta\vert \mathcal{D}\right)\vert \vert \theta-\hat{\theta}\vert \vert ^{2}d\theta\\
 & =\intop p\left(\theta\vert \mathcal{D}\right)\frac{\partial}{\partial\hat{\theta}}\vert \vert \theta-\hat{\theta}\vert \vert ^{2}d\theta\\
 & =2\intop p\left(\theta\vert \mathcal{D}\right)\hat{\theta}d\theta-2\intop p\left(\theta\vert \mathcal{D}\right)\theta d\theta\\
 & =2\left(\hat{\theta}-\mathbb{E}\left[\theta\vert \mathcal{D}\right]\right)\stackrel{!}{=}0\\
\Leftrightarrow & \hat{\theta}=\mathbb{E}\left[\theta\vert \mathcal{D}\right]
\end{align}
$$

Notice that this estimator is the best for every dataset, so it's also the one that minimizes the integral in the definition of the BMSE. In other words, the optimal estimator in terms of the BMSE is the posterior mean:

$$
\begin{equation}
\hat{\theta}_{\text{MMSE}}=\mathbb{E}\left[\theta\vert \mathcal{D}\right]
\end{equation}
$$
<span style='float:right'> $\square$ </span>

{% enddetails %}


## Other Bayes-Optimal Estimators

The MMSE estimator is frequently used but in practice any loss function can be used in order to evaluate the estimator; the squared error loss is only one choice. For instance, we can exchange the $\ell_2$ norm into any other $p$ -norm in the loss function<d-footnote>The following notation is by no means the standard, I just made it up right now.</d-footnote>:

$$
\begin{equation}
\mathcal{R}_p(\hat\theta)=\mathbb{E}_{\theta,\mathcal{D}}\left[\vert \vert \theta-\hat\theta(\mathcal{D})\vert \vert _p^p\right]
\end{equation}
$$

Under these risks, the MMSE will not be the optimal estimator. However, can again decompose the loss as we did in the proof for the MMSE:

$$
\begin{equation}
\mathcal{R}_p(\hat\theta)=\mathbb{E}_\mathcal{D}\left[J_p(\theta\vert \mathcal{D})\right]
\end{equation}
$$
where $J_p(\theta\vert \mathcal{D})=\intop p(\theta\vert \mathcal{D})\vert \vert \theta-\hat\theta\vert \vert ^p_pd\theta$ . If this function is convex and we are able to find the minimum, then the same trick as used in the MMSE proof can be used for these losses as well.


## The 0-1 Loss

Another loss function over estimators that is of interest to us is the _Bayesian 0-1 loss_, or _Bayesian binary error_ (BBE), defined by:

$$
\begin{equation}
\text{BBE}(\hat{\theta})=\mathbb{E}_{\theta,\mathcal{D}}\left[1-\delta\left(\hat\theta(\mathcal{D})-\theta\right)\right]
\end{equation}
$$
where:

$$
\begin{equation}
\delta(\hat\theta-\theta)=\cases{1\ \ \ \ \theta=\hat\theta\\0\ \ \ \ \text{otherwise}}
\end{equation}
$$

This loss gives the same penalty to every estimator that is different than the true value, regardless of how far.

Using an estimator that is optimal under this kind of loss is useful in classification. For example, assume that the estimator can have one of two values $\theta\in\\{-1, 1\\}$  and the data is a single point. In such a case, the MMSE we saw before would give a value in between -1 and 1; it will never be in the same set of values as the parameters. However, in the task of classification, we want to predict which class generated a sample. Returning "the sample is a little bit from class -1 and the rest is from class 1" is not a viable prediction. In such a case, we would want the estimator that is optimal in terms of the 0-1 loss.



### Maximum a-Posteriori Estimator

> **Theorem**:  $\theta_{\text{MAP}}=\arg\max_\theta p(\theta\vert \mathcal{D})$  is the Bayes-optimal estimator in terms of the 0-1 loss<d-footnote>Under the assumption that $p(\theta\vert \mathcal{D})$ is smooth.</d-footnote>.

{%details Click here to see the proof%}
#### Proof:

First, let's define a relaxation of the 0-1 loss which will help us prove the theorem. Define the $\epsilon$ -ball around $\theta$ as:

$$
\begin{equation}
\delta_\epsilon (\hat\theta-\theta)= \cases{1\ \ \ \ \vert \vert \theta-\hat\theta\vert \vert \le\epsilon\\0\ \ \ \ \text{otherwise}}
\end{equation}
$$

This allows us to define the $\epsilon$ -relaxed version of the $\text{BBE}$:

$$
\begin{equation}
\text{BBE}_\epsilon(\hat{\theta})=\mathbb{E}_{\theta,\mathcal{D}}\left[1-\delta_\epsilon(\hat\theta-\theta)\right]
\end{equation}
$$

Note that at the limit $\epsilon\rightarrow 0$, $\text{BBE}$ and $\text{BBE}_\epsilon$ are equal.

We will now use the same decomposition from the proof for the MMSE. In this case:

$$
\begin{equation}
\text{BBE}_\epsilon(\hat{\theta})=\mathbb{E}_{\mathcal{D}}\left[J_\epsilon(\theta\vert \mathcal{D})\right]
\end{equation}
$$

$$
\begin{align}
J_{\epsilon}\left(\theta\vert \mathcal{D}\right) & =\intop p\left(\theta\vert \mathcal{D}\right)\left(1-\delta_{\epsilon}\left(\hat{\theta}-\theta\right)\right)d\theta\\
 & =\intop p\left(\theta\vert \mathcal{D}\right)d\theta-\intop p\left(\theta\vert \mathcal{D}\right)\delta_{\epsilon}\left(\hat{\theta}-\theta\right)d\theta\\
 & =-\intop p\left(\theta\vert \mathcal{D}\right)\delta_{\epsilon}\left(\hat{\theta}-\theta\right)d\theta
\end{align}
$$

As you remember, we want to find the $\hat\theta$  that minimizes $J_{\epsilon}\left(\theta\vert \mathcal{D}\right)$ for every possible $\theta$ and $\mathcal{D}$ , which is equivalent to maximizing the term in the integral. Assuming that $p(\theta\vert \mathcal{D})$ is smooth, there exists a small $\epsilon$ around the maximum in which the function can be approximated by the constant function. In this regime of $\epsilon$ s, the $\hat{\theta}$ that maximizes the integral is the maximum of the posterior. Let $\theta_{\text{max}}=\arg\max_\theta p(\theta\vert \mathcal{D})$ , then:

$$
\begin{align}\min_{\hat{\theta}}J_{\epsilon}\left(\theta\vert \mathcal{D}\right) & =-\max_{\hat{\theta}}\intop p\left(\theta\vert \mathcal{D}\right)\delta_{\epsilon}\left(\hat{\theta}-\theta\right)d\theta\\
 & \stackrel{\epsilon\ll 1}{=}-\intop p\left(\theta\vert \mathcal{D}\right)\delta_{\epsilon}\left(\theta_{\text{max}}-\theta\right)d\theta
\end{align}
$$

Taking $\epsilon\rightarrow 0$  we get:

$$
\begin{equation}
\intop p\left(\theta\vert \mathcal{D}\right)\delta_{\epsilon}\left(\theta_{\text{max}}-\theta\right)d\theta\stackrel{\epsilon\rightarrow 0}{=}\intop p\left(\theta\vert \mathcal{D}\right)\delta\left(\theta_{\text{max}}-\theta\right)d\theta=p(\theta_\text{max}\vert \mathcal{D})
\end{equation}
$$

In words - the 0-1 loss is minimized at the point $\hat{\theta}=\theta_\text{max}$ . This means that the Bayes-optimal estimator for the 0-1 loss is given by:

$$
\begin{equation}
\hat{\theta}(\mathcal{D})=\arg\max_\theta p(\theta\vert \mathcal{D})
\end{equation}
$$

which is the maximum a-posteriori  (MAP) estimator.
<span style='float:right'> $\square$ </span>
{% enddetails %}
<br>

# Discussion

Notice that while we have shown some estimates that are optimal under different loss functions, this optimality was predicated on many assumptions. First, we assumed that the data was truly generated according to the likelihood function $p(\mathcal{D}\vert\theta)$ , which affects both Bayesian and non-Bayesian approaches. The second assumption was that we know the true prior $p(\theta)$ , which is possibly even more suspect than the first assumption.

It is important, in this case, to remember that in any modeling task the act of choosing the likelihood itself as well as the prior are subjective choices; in real life, nature probably didn't create data in the same process as any of our models. This casts the whole process of analyzing whether an estimate is optimal or not into question - what does it matter, if we can never be sure our assumptions are correct? While I think that this is definitely something to keep in mind, we would also always want to know that under _our specific assumptions_ at least our estimates are going to be the best they can be. For this reason alone, I believe that taking the time time to understand this decision-theoretic approach is worth it.

In the following posts we will start to learn about distributions and tasks where we can take this information to extract useful estimates. The first step will be to define the most simple distribution that is actually useful - the Gaussian distribution.
<br>

---

<span style='float:left'><a href="https://friedmanroy.github.io/BML/1_philosophy/">← The Bayesian Philosophy</a></span><span style='float:right'><a href="https://friedmanroy.github.io/BML/3_gaussians/">The Gaussian Distribution →</a></span>