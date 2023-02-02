---
layout: distill
comments: true
title: 7 - Evidence Function

description: The evidence function (or marginal likielihood) is one of the cornerstones of Bayesian machine learning. This post shows the construction of the evidence and how it can be used in the context of Bayesian linear regression.   

authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University

toc:
  - name: Calculating the Evidence
  - name: Evidence in Bayesian Linear Regression
  - name: Equivalent Derivation
  - name: Examples
  - name: Regarding Calibration
---

As we have previously discussed, there are many possible basis functions $h\left(\cdot\right)$ we can use to fit the linear regression model, and it is not always so simple to determine which set of basis functions is the correct one to use. On one hand, if we use a very expressive set of basis functions, or a very large one, then the model will easily fit the training data, but will probably give very inaccurate predictions for unseen data points. On the other hand, if we use a model that is too simplistic, then we will end up missing all of the data points. 

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_6/hypothesis_choosing.png"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 1: an example of the dilemma of choosing which model should be used. In both plots, a linear function and a 9th order polynomial are fitted to the data. In each case, how can we choose which of the basis functions should be used for linear regression?
</div>


This is exactly the dilemma represented in figure 1; on the left, it is fairly obvious that the straight line should be chosen, although the 9th order polynomial fits the data better. On the other hand, the graph on the right shows exactly the opposite - the 9th order polynomial intuitively looks like it explains the data better than the linear function. However, in both cases the 9th order polynomial has much higher likelihood. So how can we choose which of the basis functions is a better fit for the data?


The _evidence function_ [^1] (also called the _marginal likelihood_, since we marginalize the parameters out of the distribution) is a way for us to intelligently choose which parameterization to use. The idea behind the evidence function is to "integrate out" the specific values of the parameters $\theta$ and to see how probable the data set is under our parameterization. Suppose we have a prior $p\left(\theta\mid \Psi\right)$ that is dependent
on some parameters $\Psi$ . Then:

$$
\begin{align}
p\left(\mathcal{D}\mid \Psi\right) & =\intop p\left(\mathcal{D},\theta\mid \Psi\right)d\theta\nonumber \\
 & =\intop\underbrace{p\left(\mathcal{D}\mid \theta\right)}_{\text{likelihood}}\cdot\underbrace{p\left(\theta\mid \Psi\right)}_{\text{prior}}d\theta
\end{align}
$$

The way it is written at the moment may be a bit confusing. Up until
now (and from now on, as well), we wrote the prior as $p\left(\theta\right)$ ,
but suddenly we're adding the conditioning on $\Psi$ - why? When
we define a prior, we usually have to choose a distribution for the
prior. Often, this distribution has parameters that define it; for
instance, if the prior is a Gaussian, then the parameters are the
specific $\mu_{0}$ and $\Sigma_{0}$ we chose. In our new notation
$\Psi=\\{ \mu_{0},\Sigma_{0}\\}$ , and we want to compare
between different possible $\Psi$ s.

---

#### Simple Bayesian linear regression

As an example, suppose we assume that:

$$
\begin{align}
y & =\theta x
\end{align}
$$

Furthermore, we assume that we have two priors on $\theta$ given by:

$$
\begin{align}
\theta & \sim\mathcal{N}\left(\mu_{1},\Sigma_{1}\right)\\
\tilde{\theta} & \sim\mathcal{N}\left(\mu_{2},\Sigma_{2}\right)
\end{align}
$$

In other words, we have two competing priors, from which we want to choose only one. We can then calculate:

$$
\begin{equation}
p\left(y\mid \mu_{i},\Sigma_{i}\right)=\intop p\left(y\mid \theta\right)p\left(\theta\mid \mu_{i},\Sigma_{i}\right)d\theta
\end{equation}
$$

for $i\in\\{ 1,2\\}$ . If we calculate this probability for both $\mu_{1}$ and $\mu_{2}$ , then we will get a value that tells us how likely the training data $y$ is under each of these different assumptions. That is, instead of asking how probable $y$ is under a specific value of $\theta$ (which is just the likelihood), this is like asking how probable $y$ is when averaging out the values of $\theta$ , _given the parameterization_ of $\mu_{i}$ and $\Sigma_{i}$ .

Of course, the priors could assume different basis functions, as in:

$$
\begin{align}
f_{1}\left(x\right) & =\theta x\qquad\theta\sim\mathcal{N}\left(\mu_{\theta},\Sigma_{\theta}\right)\\
f_{2}\left(x\right) & =\beta_{0}+\beta_{1}x\qquad\left(\begin{matrix}\beta_{0}\\
\beta_{1}
\end{matrix}\right)\sim\mathcal{N}\left(\mu_{\beta},\Sigma_{\beta}\right)
\end{align}
$$

and we want to choose between $\Psi_{\theta}=\\{ \mu_{\theta},\Sigma_{\theta}\\}$ and $\Psi_{\beta}=\\{ \mu_{\beta},\Sigma_{\beta}\\}$ . Notice that the basis functions we assume are themselves compared when we do this, simply since the basis functions are part of the assumptions we made when we chose our prior.

---

# Calculating the Evidence

Suppose that our prior is described, as above, by:

$$
\begin{equation}
\theta\sim p\left(\theta\mid \ \Psi\right)
\end{equation}

$$
where $\Psi$ are some parameters. We want to find the value of $p\left(\mathcal{D}\mid \ \Psi\right)$ - the evidence for seeing the data under this parameterization $\Psi$ . Recall from Bayes' law:

$$
\begin{align}
p\left(\theta\mid \ \mathcal{D},\Psi\right) & =\frac{p\left(\theta,\mathcal{D}\mid \Psi\right)}{p\left(\mathcal{D}\mid \Psi\right)}\\
\Leftrightarrow p\left(\mathcal{D}\mid \Psi\right) & =\frac{p\left(\theta,\mathcal{D}\mid \Psi\right)}{p\left(\theta\mid \ \mathcal{D},\Psi\right)}\\
\Leftrightarrow p\left(\mathcal{D}\mid \Psi\right) & =\frac{p\left(\mathcal{D}\mid \theta\right)p\left(\theta\mid \Psi\right)}{p\left(\theta\mid \ \mathcal{D},\Psi\right)}
\end{align}
$$

This is true for _every_ choice of $\theta$ ! A common way to find the evidence function is by plugging $\hat{\theta}_{\text{MAP}}$ into the expression, which gives:

$$
\begin{equation}\label{eq:general-evidence}
\Leftrightarrow p\left(\mathcal{D}\mid \Psi\right)=\frac{p\left(\mathcal{D}\mid \hat{\theta}_{\text{MAP}}\right)p\left(\hat{\theta}_{\text{MAP}}\mid \Psi\right)}{p\left(\hat{\theta}_{\text{MAP}}\mid \ \mathcal{D},\alpha\right)}
\end{equation}
$$

Usually, the numerator is either known or pretty simple to calculate, while the denominator is quite hard to find. In such cases, the denominator is approximated in some manner in other to find the evidence. Luckily for us, the denominator is easy to calculate in the case of Bayesian linear regression with a Gaussian prior. 

# Evidence in Bayesian Linear Regression

In standard Bayesian linear regression, the posterior is a Gaussian. We can utilize this knowledge to find a more specific formula for the evidence. Notice that:

$$
\begin{align}
p\left(\hat{\theta}_{\text{MAP}}\mid \ y,\Psi\right) & =\max_{\theta}p\left(\theta\mid \ \mathcal{D},\Psi\right)\\
 & =\max_{\theta}\frac{1}{\sqrt{\left(2\pi\right)^{d}\mid \Sigma_{\theta\mid \mathcal{D}}\mid }}e^{-\frac{1}{2}\left(\theta-\mu_{\theta\mid \mathcal{D}}\right)^{T}\Sigma_{\theta\mid \mathcal{D}}^{-1}\left(\theta-\mu_{\theta\mid \mathcal{D}}\right)}
\end{align}
$$

This maximum is attained at $\theta=\mu_{\theta\mid \mathcal{D}}$ , where the whole term in the exponent is equal to 1, so we're left with:

$$
\begin{equation}
p\left(\hat{\theta}_{\text{MAP}}\mid \ y,\Psi\right)=\frac{1}{\sqrt{\left(2\pi\right)^{d}\mid \Sigma_{\theta\mid \mathcal{D}}\mid }}
\end{equation}
$$

Plugging this into equation \eqref{eq:general-evidence}, we have:

$$
\begin{equation}\label{eq:evidence-function}
p\left(y\mid \Psi\right)=\left(2\pi\right)^{N/2}\mid \Sigma_{\theta\mid \mathcal{D}}\mid ^{1/2}p\left(y\mid \hat{\theta}_{\text{MAP}}\right)p\left(\hat{\theta}_{\text{MAP}}\mid \Psi\right)
\end{equation}
$$

But we can be even more specific by plugging in the MAP estimate under a Gaussian prior:

$$
\begin{equation}
p\left(y\mid \mu,\Sigma\right)=\left(2\pi\right)^{N/2}\mid \Sigma_{\theta\mid \mathcal{D}}\mid ^{1/2}\mathcal{N}\left(\mu_{\theta\mid \mathcal{D}}\mid \;\mu,\Sigma\right)\mathcal{N}\left(y\mid \;H\mu_{\theta\mid \mathcal{D}},I\sigma^{2}\right)
\end{equation}
$$


# Equivalent Derivation

The above derivation allows us to calculate the actual value of the evidence quickly, but it may be a bit harder to understand what is going on in this form. An equivalent way to find the evidence is to find $p\left(\mathcal{D}\mid \Psi\right)$ directly, from the definition. Recall that we modeled linear regression according to:

$$
\begin{equation}
y=H\theta+\eta\quad\eta\sim\mathcal{N}\left(0,I\sigma^{2}\right)
\end{equation}
$$

If we marginalize $\theta$ out of the above equation, it will still be an exponent of something that is quadratic in $y$, so it will be a Gaussian. So we just need to find the mean and covariance in order to find the exact form of the Gaussian:

$$
\begin{align}
\mathbb{E}[y] & =\mathbb{E}[H\theta+\eta] \\
 & =H\mathbb{E}[\theta]+\mathbb{E}[\eta]\nonumber \\
 & =H\mu_0
\end{align}
$$

The expectation is always the easiest part, but in this case the covariance isn't much harder to find:
$$
\begin{align}
\text{cov}[y]&=\text{cov}[H\theta+\eta]\\&=\text{cov}[H\theta]+\text{cov}[\eta]\\&=H\text{cov}[\theta]H^T+I\sigma^2\\&=H\Sigma_0H^T+I\sigma^2
\end{align}
$$

So, the evidence function for Bayesian linear regression is actually the density of the following Gaussian at the point $y$ :

$$
\begin{equation}
p\left(y\mid \mu_0,\Sigma_0\right)=\mathcal{N}\left(y\,\mid \,H\mu_0,\;H\Sigma_0 H^{T}+I\sigma^{2}\right)
\end{equation}
$$

---
#### Example: Learning the sample noise

Notice that all this time we assumed that we know the variance of the sample noise, $\sigma^{2}$ . This really helps simplify many of the derivations we made, but is kind of a weird assumption to make. 

We can try to use a fully Bayesian approach, where we choose a prior for $\sigma^{2}$ and then calculate the posterior. If we try to choose a Gaussian as a prior, we quickly run into a problem - $\sigma^{2}$ can't be negative, but every Gaussian will have a positive density at negative values! In addition, the Gaussian distribution is symmetric, but the distribution we want to describe $\sigma^{2}$ with is probably very asymmetrical, with low density at values close to zero, high density later on, and a long tail for higher values. So, clearly we can't use a Gaussian as the prior for $\sigma^{2}$ . There are distributions that match the above description, but we haven't discussed them (and won't). Also, finding their posterior is usually a bit harder than finding the posterior of a Gaussian distribution[^2] . So, going fully Bayesian is more complicated in this case.

Instead, we can use the evidence function in order to choose the most fitting sample noise. In the notation above, we only wrote $y$ as a function of $\mu$ and $\Sigma$ , but it is obviously affected by $\sigma^{2}$ through the covariance:

$$
\begin{equation}
\text{cov}[y]=H\Sigma_0 H^{T}+I\sigma^{2}
\end{equation}
$$

We can define the evidence as a function of the variance as well and then choose from a closed set of values chosen ahead of time $S=\\{ \sigma_{i}^{2}\\}_{i=1}^{q}$ , in which case we would say:

$$
\begin{equation}
\hat{\sigma}^{2}=\arg\max_{\sigma^{2}\in S}\mathcal{N}\left(y\,\mid \,H\mu_0,\;H\Sigma_0 H^{T}+I\sigma^{2}\right)
\end{equation}
$$

Another option is to use gradient ascent (or another optimization algorithm) in order to find the maximum iteratively:

$$
\begin{equation}
\hat{\sigma}_{\left(t\right)}^{2}=\hat{\sigma}_{\left(t-1\right)}^{2}+\epsilon\nabla_{\sigma}\log p\left(y\,\mid \,\mu_0,\Sigma_0,\sigma\right)
\end{equation}
$$

where $\epsilon$ is some learning rate. However, note that there is no guarantee that the evidence is a concave function!

---

# Examples

The way evidence is presented is usually not very intuitive. Let's look at the definition of the evidence for a second:

$$
\begin{equation}
p(\mathcal{D}\mid \Psi)=\intop p(\mathcal{D}\mid \theta) p(\theta\mid \Psi)d\theta =\mathbb{E}_{\theta\mid \Psi}\left[p(\mathcal{D}\mid \theta)\right]
\end{equation}
$$

This is the most direct (and intractable) way to define the evidence, but is a bit more approachable in an abstract way. Notice that the operation we actually have here is (basically) a sum of all possible likelihoods of fits according to the prior (the $p(\mathcal{D}\mid \theta)$ ), weighted by the prior probability. This means that if the data has high likelihood under the values of the prior with high density, then the evidence for the prior will be high. On the other hand, if the area with highest density on the prior isn't even close to the data, then the evidence will be low. This is shown in the figure below[^3].

<div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_6/evidence_fig.png"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 2: a graphical plot for two models - one with high evidence, and one with low evidence. The red dots are the given data, which are the same for the left and right plots. The gray contour lines represent the prior's density over possible functions while the black line is the mean of the prior.
</div>

#### Changing the basis functions

Evidence can be used to decide which basis function to use. At the beginning of this post, I showed an example with two polynomials, but we can compare more than just two functions each time. 

Below, I show the same as figure 2, just for Gaussian basis functions which are defined according to:

$$
\begin{equation}
h_i(x)=e^{-\frac{\mid \mid x-\mu_i\mid \mid ^2}{2\beta}}
\end{equation}
$$
 for some fixed $\beta$ :
 
 <div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_6/rbf_evidence_unsymm.gif"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 3: a visualization of the evidence for many different choices of $\mu_1$ and $\mu_2$ . On the left, the red dots are the data points while the gray area and black line are the prior density. On the right is the contour plot of the evidence for pairs of values $(\mu_1, \mu_2)$ - the black dot defines which value of both to display on the left. The white number under the moving dot is the log-evidence for that specific point in parameter-space.
</div>

The chosen prior for this example is:

$$
\begin{equation}
\theta\sim\mathcal{N}\left(\left[\matrix{a\\-a}\right],\ I\alpha^2\right)
\end{equation}
$$
for some value of $a$ and $\alpha$ .

Note that while the evidence space looks symmetric, it isn't exactly so. The evidence for the lower-right corner is slightly higher than that of the upper-right corner. The reason for this is because of the choice of prior, since the defined function is:

$$
\begin{equation}
f_\theta(x)=\theta_1 h_1(x)+ \theta_2h_2(x)+\eta
\end{equation}
$$


#### Changing the Prior Mean

Evidence can also be used to choose which prior is suitable. For instance, suppose that our prior is given by:

$$
\begin{equation}
\left(\matrix{\theta_0\\ \theta_1}\right)\sim \mathcal{N}\left(\left(\matrix{\mathbb{E}[\theta_0]\\ \mathbb{E}[\theta_1]}\right),\ \ I\alpha^2\right)
\end{equation}
$$
where the function is modeled as $f_\theta(x)=\theta_0 + \theta_1 x+\eta$ . The animation below depicts how the evidence changes when the means $\mathbb{E}[\theta_i]$ are changed:


 <div class="fake-img l-page">
<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_6/lin_evidence.gif"  
style="display: inline-block; margin: 0 auto; ">
</p>
</div>
<div class="caption">
    Figure 4: the same as the previous visualization, only for linear functions. In this animation, the size of the dot corresponds to the sample-specific likelihood; notice how all points are large at the maximum evidence.
</div>

---

# Regarding Calibration

The evidence is certainly a useful tool for model selection, but it should be used carefully. In particular, relying too much on the evidence without taking care can result in overfitting to the training data. More concisely, high evidence doesn't equal to good generalization! This is because, by design, the marginal likelihood gives high scores to priors which explain the data well. 

This is really problematic if we don't take care to separate the manner in which we choose the prior from the model selection stage. For instance, assume we have some data $\mathcal{D}$ and want to find the model that gives this data the highest possible evidence. Recall that:

$$
\begin{equation}
p\left(y\mid \Psi\right)=\mathcal{N}\left(y\,\mid \,H\mu_0,\;H\Sigma_0 H^{T}+I\sigma^{2}\right)
\end{equation}
$$
In such a case, we can **always** define the basis functions/prior such that $H\mu_0=y$  , $\Sigma_0=0$  and $\sigma^2$ is arbitrarily small. This will result in the following, not very helpful, evidence:

$$
\begin{equation}
p\left(\tilde{y}\mid \Psi\right)=\delta(y)
\end{equation}
$$
which is equal to infinity when $\tilde{y}=y$ , a really good evidence score. However, this is obviously not the model we would want to choose! Such a model will not generalize to any new data, which is basically the definition of overfitting.

While the above is kind of a silly case we won't see in real life, it still illustrates the problems that may arise when optimizing the evidence. Specifically, if the set of possible models contains many very specific and expressive models, then optimizing on this set of models is prone to overfitting. On the other hand, it could be tempting to iterate over different sets of possible models as we see the results; again, this would just lead to models that are overfit and won't be calibrated towards new (unseen) data points. 


### Mitigation

While the above is true, we can take steps to ensure this doesn't happen. For starters, the discussion above is kind of backwards of how we defined the fitting process in the first place; we assumed that we have several priors that (we believe) explain the data equally well, _and only then_ do we want to select one out of these possible priors. 

In other words, if we remain true to the original Bayesian method and choose a set of priors **before we ever see the data**. Only then would we try to select  one of them, then the biasing towards the data that was described above is mitigated, in the sense that the selection of the priors is independent of the data. This, as mentioned, is more in line with the Bayesian philosophy.

Another thing to keep in mind is that maximizing the evidence will probably work best when the set of hypotheses is small; this is to ensure that we don't allow too fine-grained definitions of priors.


### Priors all the Way Down

Probably the "true Bayesian method" to overcome these problems is a different approach all together. In the first place, assuming that all these sets of hypotheses we were talking about are equally likely is questionable. Instead, we would want to choose some sort of _hyperprior_ - a prior over the hyperparameters:

$$
\psi\sim p(\Psi\mid \xi)
$$

where $\xi$  are the parameters of the hyperprior. 

In such a setting, our posterior distribution will be defined by _integrating out_ the hyperparameters:

$$
p(\theta\mid \mathcal{D})=\intop p(\theta\mid \Psi,\mathcal{D})p(\Psi\mid \xi)d\Psi
$$

This is nice since it bypasses the need to choose a model - simply integrate over all of them, a more Bayesian approach. It also incorporates are beliefs explicitly - in the integral above we explicitly assumed that $\Psi$ is independent of $\mathcal{D}$ !

That said, we have introduced a new complication; how should $\xi$  be chosen? Continuing with this reasoning, shouldn't we also incorporate a prior over $\xi$, a so called "hyper-hyperprior",  and so on? While these concerns are valid, the "priors all the way down" kind of approach typically stops at the hyperprior, since it is far enough from the data term.

---

[^1]: Bishop section 3.4 or [MacKay's chapter on model selection in "Information Theory, Inference and Learning Algorithms"](http://www.inference.org.uk/mackay/itprnn/ps/343.355.pdf)
[^2]: If you are curious, you can look at Bishop section 2.3.6 for a full derivation of the posterior under a proper prior.
[^3]: The way the prior is displayed in these plots is by calculating the mean and standard deviations of $\mathcal{N}\left(H\mu_0,\ H\Sigma_0 H^T + I\sigma^2\right)$ for every point in space, which is exactly like using the equivalent definition of the evidence.
