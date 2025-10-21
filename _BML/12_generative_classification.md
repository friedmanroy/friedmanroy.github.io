---
layout: distill
comments: false
title: Generative Classification
description: Discriminative classification, while being simple, is also quite hard to treat in the Bayesian framework. Generative classification is slightly more forgiving, and is the focus of this post.
date: 2024-03-01
authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University
toc:
  - name: Bayes-Optimal Classification
  - name: Classes as Gaussians
  - name: Fitting the Models
  - name: Discussion
---

<span style="float:left"><a href="https://friedmanroy.github.io/BML/11_discriminative_classification/">← Discriminative classification</a></span><span style="float:right"><a href="https://friedmanroy.github.io/BML/13_gmms/">GMMs and EM →</a></span>
<br>
<br>

> Last post we talked about discriminative classification, which was a straightforward manipulation of the task we've seen in linear regression into classification. However, the integer nature of the problem added problems that didn't exist in the world of linear regression. In this post, we move to a different framework of classification

In discriminative classification, we attempted to define the process that turns a data point $x$ into a label $y$. That is, the following probability is estimated:

$$
\begin{equation}
p(y=c|x,\theta)=?
\end{equation}
$$

However, if we have a good estimate of the distribution $p(x\vert y=c)$, then we can do better.

<br>

# Bayes-Optimal Classification

The usual loss function considered for classification, in most settings, is the 0-1 loss:

$$
\begin{equation}
\ell(y,\hat{y})=\begin{cases}1&y=\hat{y}\\0&y\neq\hat{y}\end{cases}
\end{equation}
$$

In <a href="https://friedmanroy.github.io/BML/2_estimates/">the post on Bayesian decision theory</a>, we saw that the _optimal_ estimator in this setting is the MAP estimate. That is, if we have direct access to the distribution $p(y=c)$ _and_ we know the likelihood $p(x\vert y=c)$, then (on average) the best classifier is:

$$
\hat{y}(x) = \arg\max_c p(y=c|x)=\arg\max_c p(x|y=c)p(y=c)
$$

Generative classification attempts to utilize this in order to get close to optimal performance. To do this, a set of parameters $\Phi$ are used in order to define a parametric model that estimates $p(y=c\vert\Phi)\approx p(y=c)$ and $p(x\vert y=c,\Phi)\approx p(x\vert y=c)$. If we are able to model these distributions well, then it is ensured that performance will be close to optimal.

### Our Assumptions 

Before continuing, we should take a closer look at our assumptions. The assumptions for classification that are assumed above are:

1. The 0-1 loss is what we're interested in minimizing
2. Our estimate $p(y=c\vert \Phi)$ is accurate
3. The prior defined for each class $p(x\vert y=c,\Phi)$ matches the data distribution well

But do these assumptions make any sense? Well:
1. The 0-1 loss is the one most used in literature (also for discriminative classification). Is it justified? Well, in some cases it might be justified, but sometimes it doesn't make much sense. For instance, if we have a model that classifies between species of animals given some features, it makes sense that confusing a dog's breed should be penalized less than mixing up a dog and an elephant. In another example, suppose you're using a classification model for self-driving cars; would you want to use the same error when the model classifies a person as a drivable part of road v.s. a drivable part of road as a person?
2. If the distribution of the classes we view in the real world is basically balanced, then estimating $p(y=c)$ should be quite straightforward. However, this might not always be the case, and $p(y=c\vert\Phi)$ might be off
3. The assumption that $p(x\vert y=c,\Phi)$ models the true data distribution is, honestly, the least plausible assumption. When $x$ is very high-dimensional, actually capturing the distribution is almost completely infeasible

### Why Generative?

So, our earlier assumptions are really suspect. 

However, generative classification is still a viable method that is definitely usable. First, unlike discriminative classification, _our prior knowledge can be directly utilized_. If we truly know something about the process that created the data in each class, we can directly encode this in the classifier. 

As an example, we know something about the physical process for rain to fall from the sky. Given the pressure, temperature, humidity etc. we can then explicitly define a (maybe unnormalized) probability that rain should fall given these values and our understanding of physics. From past data, we also have some intuition for the overall chance of rain any day of the year. These two things, together, can be used to classify whether there will be rain today or not.

Finally, if we have access to $p(x\vert y=c,\Phi)$ we can actually solve other problems as well such as _signal restoration_. That is, _if $p(x\vert y=c,\Phi)$ is a good placeholder for the true distribution, we can solve many tasks at the same time, without training a new model for each task_. 

Optimality, together the fact that we can use this distribution to solve other problems, is the driving force behind generative modeling. 

<br>

# Classes as Gaussians

The simplest model for these class-specific densities is using a Gaussian for each class:

$$
\begin{equation}
p\left(x|y=c,\Phi\right)=\mathcal{N}\left(x\,|\,\mu_{c},\Sigma_{c}\right)
\end{equation}
$$

where $\mu_{c}$ and $\Sigma_{c}$ are the class specific parameters:

Very broadly, whenever Gaussians are used to model the class-specific densities, then this technique is called _Gaussian discriminant analysis_ (GDA). The distribution as a whole will be:

$$
\begin{align}
p\left(x|\Phi\right) & =\sum_{c}^{C}p\left(x|y=c,\Phi\right)p\left(y=c|\Phi\right) \\
 & =\sum_{c}^{C}p\left(y=c|\Phi\right)\mathcal{N}\left(x\,|\,\mu_{c},\Sigma_{c}\right)\\
 & =\sum_{c}^{C}\pi_{c}\mathcal{N}\left(x\,|\,\mu_{c},\Sigma_{c}\right)
\end{align}
$$

where, again, $y$ gives us a label for the class. We also defined $p\left(y=c\vert\Phi\right)\stackrel{\Delta}{=}\pi_{c}$ for slightly shorter notation. Overall, our parameters are:

$$
\begin{equation}
\Phi=\left\{\pi_1,\mu_1,\Sigma_1,\cdots,\pi_C,\mu_C,\Sigma_C\right\}
\end{equation}
$$

The probability that a new sample $x$ belongs to the class $c$ is given by:

$$
\begin{align}
p\left(y=c|x,\Phi\right) & =\frac{p\left(x,y=c|\Phi\right)}{p\left(x|\Phi\right)} \\
 & =\frac{p\left(x|y=c,\Phi\right)p\left(y=c|\Phi\right)}{\sum_{c'}p\left(x|y=c',\Phi\right)p\left(y=c'|\Phi\right)}\\
 & =\frac{\pi_{c}\mathcal{N}\left(x\,|\mu_{c},\Sigma_{c}\right)}{\sum_{c'}\pi_{c'}\mathcal{N}\left(x\,|\mu_{c'},\Sigma_{c'}\right)}
\end{align}
$$

So, after we find the parameters $\pi_{c}$, $\mu_{c}$ and $\Sigma_{c}$, we can classify a new point $x$ using:

$$
\begin{align}
\hat{y} & =\arg\max_{c}\left[p\left(y=c|x,\Phi\right)\right]\\
 & =\arg\max_{c}\left[\frac{\pi_{c}\mathcal{N}\left(x\,|\thinspace\mu_{c},\Sigma_{c}\right)}{\sum_{c'}\pi_{c'}\mathcal{N}\left(x\,|\thinspace\mu_{c'},\Sigma_{c'}\right)}\right]\\
 & =\arg\max_{c}\left[\pi_{c}\mathcal{N}\left(x\,|\thinspace\mu_{c},\Sigma_{c}\right)\right]\\
 & =\arg\max_{c}\left[\log\left(\pi_{c}\mathcal{N}\left(x\,|\thinspace\mu_{c},\Sigma_{c}\right)\right)\right]\\
 & =\arg\max_{c}\left[\log\pi_{c}+\log\mathcal{N}\left(x\,|\thinspace\mu_{c},\Sigma_{c}\right)\right]
\end{align}
$$


## Linear Discriminant Analysis (LDA)

Before we look at the general case, let's look at the simpler case where the covariance matrices are shared across all classes $\Sigma_{c}=\Sigma\stackrel{\Delta}{=}\Lambda^{-1}$. We can now slightly simplify the class conditional distribution:

$$
\begin{align}
p\left(y=c|x,\theta\right) & =\frac{1}{Z}\frac{1}{C}\pi_{c}\exp\left[-\frac{1}{2}\left(x-\mu_{c}\right)^{T}\Lambda\left(x-\mu_{c}\right)\right]\\
 & =\frac{1}{Z}\frac{1}{C}\pi_{c}\exp\left[x^{T}\Lambda\mu_{c}-\frac{1}{2}\mu_{c}^{T}\Lambda\mu_{c}-\frac{1}{2}x^{T}\Lambda x\right]\\
 & =\frac{1}{C}\frac{\pi_{c}\exp\left[x^{T}\Lambda\mu_{c}-\frac{1}{2}\mu_{c}^{T}\Lambda\mu_{c}\right]\exp\left[-\frac{1}{2}x^{T}\Lambda x\right]}{\sum_{c'}\frac{1}{C}\pi_{c'}\exp\left[x^{T}\Lambda\mu_{c'}-\frac{1}{2}\mu_{c'}^{T}\Lambda\mu_{c'}\right]\exp\left[-\frac{1}{2}x^{T}\Lambda x\right]}\\
 & =\frac{\pi_{c}\exp\left[x^{T}\Lambda\mu_{c}-\frac{1}{2}\mu_{c}^{T}\Lambda\mu_{c}\right]}{\sum_{c'}\pi_{c'}\exp\left[x^{T}\Lambda\mu_{c'}-\frac{1}{2}\mu_{c'}^{T}\Lambda\mu_{c'}\right]}
\end{align}
$$

where $\Lambda$ is the precision matrix as we've defined before. Now, let's define $b_{c}=\log\pi_{c}-\frac{1}{2}\mu_{c}^{T}\Lambda\mu_{c}$ and $w_{c}=\Lambda\mu_{c}$. Now the above takes the form:

$$
\begin{equation}
p\left(y=c|x,\theta\right)\propto\exp\left[w_{c}^{T}x+b_{c}\right]
\end{equation}
$$


Let's look again at the classification of a new point, given by:

$$
\begin{align}
\hat{y} & =\arg\max_{c}\left[p\left(y=c|x,\Phi\right)\right]\\
 & =\arg\max_{c}\left[\log p\left(y=c|x,\Phi\right)\right]\\
 & =\arg\max_{c}\left[w_{c}^{T}x+b_{c}\right]
\end{align}
$$

Now we can clearly see that the term $w_{c}^{T}x+\gamma_{c}$ is linear! 

If we look at only two classes, we can get a sense of how this model decides which class a new data point belongs to. If we equate the conditional probabilities for the two classes, we get:

$$
\begin{align}
\log p\left(y=c|x,\Phi\right) & =\log p\left(y=c'|x,\Phi\right)\\
\Rightarrow w_{c}^{T}x+b_{c} & =w_{c'}^{T}x+b_{c'}\nonumber \\
\Rightarrow\left(w_{c}-w_{c'}\right)^{T}x & =b_{c'}-b_{c}
\end{align}
$$

So, on the line $f\left(x\right)=\left(w_{c}-w_{c'}\right)^{T}x-\left(b_{c'}-b_{c}\right)$, the probabilities for both classes are equal. We'll call this line the _decision boundary_, for obvious reasons. The decision boundary between the two Gaussians is a line, hence the name _linear discriminant_ analysis<d-footnote>Actually, this name is kind of confusing. Even though it's called discriminant analysis, this is actually a generative classification approach. These are weird naming conventions, like logistic regression for a classification task.</d-footnote>. Notice that in the same way, if we had more classes, the decision boundary between every two classes will also have this form; it'll be linear.

## Quadratic Discriminant Analysis (QDA)

Now, let us generalize to class specific covariances. Let's look at the binary case, again:

$$
\begin{align}
\log p\left(y=c_{1}|x,\Phi\right) & =\log p\left(y=c_{2}|x,\Phi\right)\\
\Rightarrow\log\pi_{1}+\log\mathcal{N}\left(x\,|\thinspace\mu_{1},\Sigma_{1}\right) & =\log\pi_{2}+\log\mathcal{N}\left(x\,|\thinspace\mu_{2},\Sigma_{2}\right)\\
\Rightarrow x^{T}\left(\Sigma_{1}^{-1}-\Sigma_{2}^{-1}\right)x- & 2x^{T}\left(\Sigma_{1}^{-1}\mu_{1}-\Sigma_{2}^{-1}\mu_{2}\right)=C
\end{align}
$$

where $C$ is the aggregation of all terms that are constant with respect to $x$. We see that the decision boundary between $c_{1}$ and $c_{2}$ is _quadratic_ in this case. This means that if we were to look at this in 2D, the decision boundary between any two classes would be a parabola. This also explains the name: _quadratic_ discriminant analysis.

<br>

# Fitting the Models

So far, we only talked about the _setting_ of generative classification and how to fit the models in practice. 

I'm going to assume that the model for _all of the classes together_ is given by:

$$
\begin{align}
p(x|\Phi)&=\sum_{c=1}^Cp(x|y=c,\Phi)p(y=c|\Phi)\\
         &=\sum_{c=1}^{C}\pi_c\cdot p(x|y=c,\Phi_c)
\end{align}
$$

So, for notational ease, denote $\pi_c=p(y=c\vert\Phi)$. Additionally, we'll assume class-specific parameters and use $\Phi_c$ to denote these class-conditional parameters. In total, are set of parameters is:

$$
\begin{equation}
\Phi=\{\pi_1,\cdots,\pi_C,\Phi_1,\cdots,\Phi_C\}
\end{equation}
$$


We now have two choices - frequentist or Bayesian estimation of $\Phi$. Because we defined the $\Phi_c$s to be class-conditional, we can simply fit each $\Phi_c$ on data points from the $c$-th class. This can be done either using MLE or by finding the posterior for these parameters:

$$
\begin{equation}
p(\Phi|\mathcal{D})=\prod_{c=1}^C p(\Phi_c|\ \{x_i|y_i=c\})
\end{equation}
$$

Analogously, the $\pi_c$s can also be fit using MLE or by defining a prior and calculating the posterior.

{% details MLE for $\pi_c$ %}
The values of $\pi_1,\cdots,\pi_C$  are constrained to be non-negative and also to sum to one. This means we'll need to use Lagrange multipliers in order to find the vector $\pi=\left(\pi_1,\cdots,\pi_C\right)^T$.

First, the constraint on $\pi_{c}$ is:

$$
\begin{align*}
\sum_{c}\pi_{c} & =1\\
\sum_{c}\pi_{c}-1 & =0\\
\Rightarrow g\left(\pi_{c}\right)= & \sum_{c}\pi_{c}-1
\end{align*}
$$

and our Lagrangian is:

$$
\begin{equation}
\mathcal{L}=\ell\left(X|\theta\right)-\lambda\left(\sum_{c}\pi_{c}-1\right)
\end{equation}
$$

Let's derive this by $\pi_{c}$:

$$
\begin{align}
\frac{\partial}{\partial\pi_{c}}\mathcal{L} & =\frac{\partial}{\partial\pi_{c}}\ell\left(X|\theta\right)-\lambda\nonumber \\
 & =\sum_{i}\mathbb{I}\left[y_{i}=c\right]\frac{\partial}{\partial\pi_{c}}\log\pi_{c}-\lambda\nonumber \\
 & =\frac{1}{\pi_{c}}\sum_{i}\boldsymbol{1}\left[y_{i}=c\right]-\lambda\nonumber \\
 & =\frac{1}{\pi_{c}}N_{c}-\lambda\stackrel{!}{=}0\nonumber \\
\Rightarrow\pi_{c} & =\frac{N_{c}}{\lambda}
\end{align}
$$

Now, we can find $\lambda$ by looking at the constraint and replacing $\pi_{c}$ by the value in the above equation:

$$
\begin{align*}
\sum_{c}\pi_{c} & =\frac{1}{\lambda}\sum_{c}N_{c}=\frac{N}{\lambda}=1\\
\Rightarrow & \lambda=N
\end{align*}
$$

Substituting back, we get:

$$
\begin{equation}
\hat{\pi}_{c}^{\text{ML}}=\frac{N_{c}}{N}
\end{equation}
$$

which makes a lot of sense.

{% enddetails %}

## Natural Prior for $\pi_c$

Looking again at our data, we see that $y_{i}\in\\{ 1,\cdots,C\\}$ and that $y_{i}$ equals $c$ with probability $\pi_{c}$; this is like saying that the likelihood of $y_{i}$ is a _categorical distribution_:

$$
\begin{equation}
y_{i}\sim\text{Categorical}\left(\pi_{1},\cdots,\pi_{C}\right)
\end{equation}
$$

As we have seen before, this means that the likelihood over $y$ is given by:

$$
\begin{equation}
p\left(y|\pi\right)=\prod_{i=1}^{N}\pi_{y_{i}}
\end{equation}
$$

Define (again) $N_{c}$ to be the number of $y_{i}$s in class $c$:

$$
\begin{equation}
N_{c}=\sum_{i=1}^{N}\boldsymbol{1}\left[y_{i}=c\right]
\end{equation}
$$

Using this definition, we can rewrite the likelihood over $y$ as:

$$
\begin{equation}
p\left(y|\pi\right)=\prod_{c}\pi_{c}^{N_{c}}
\end{equation}
$$

So far so good. We will now try to find a prior that is conjugate to this likelihood. In particular, this will be helpful because then we will know that the posterior will always be from the same class of distributions as the prior. In this case, notice that if we define the prior as:

$$
\begin{equation}
p\left(\pi\right)\propto\prod_{c}\pi_{c}^{\alpha_{c}-1}\qquad\forall\pi_{i}\;0\le\pi_{i}\le1\qquad\sum_{i}\pi_{i}=1
\end{equation}
$$

where $\forall c\;\alpha_{c}>0$, then multiplying the prior with the likelihood lands us in the same kind of distribution:

$$
\begin{equation}
p\left(\pi\right)p\left(y|\pi\right)\propto\prod_{c}\pi_{c}^{\alpha_{c}-1}\prod_{c}\pi_{c}^{N_{c}}=\prod_{c}\pi_{c}^{\alpha_{c}+N_{c}-1}=\prod_{c}\pi_{c}^{\tilde{\alpha}_{c}-1}
\end{equation}
$$

In other words, the prior turned out to be conjugate to the categorical likelihood over $y_{i}$s! As it happens, this distribution has a name - it is called the _Dirichlet distribution_ over the random vector $\pi=\left(\pi_{1},\cdots,\pi_{C}\right)^{T}$:

$$
\begin{equation}
p\left(\pi\right)\propto\prod_{c}\pi_{c}^{\alpha_{c}-1}\Leftrightarrow\pi\sim\text{Dirichlet}\left(\alpha_{1},\cdots,\alpha_{C}\right)
\end{equation}
$$

### Properties of the Dirichlet Prior

The way we built the Dirichlet distribution is as a prior for categorical distributions. If we want to consider only the marginal distribution for $\pi_{i}$, we can understand it in terms of one-vs-all. In other words, either the class belongs to $\pi_{i}$ or it doesn't, in which case we can rewrite the distribution as:

$$
\begin{equation}
p\left(\pi_{i}\right)\propto\pi_{i}^{\alpha_{i}-1}\cdot\left(1-\pi_{i}\right)^{\overline{\alpha}_{i}-1}
\end{equation}
$$

where $\overline{\alpha}\_i=\sum_{j}\alpha_{j}-\alpha_{i}$. This distribution happens to be called the _Beta distribution_:

$$
\begin{equation}
\pi_{i}\sim\text{Beta}\left(\alpha_{i},\overline{\alpha}_{i}\right)
\end{equation}
$$

The mean of the Beta distribution is given by:

$$
\begin{equation}
\mathbb{E}\left[\pi_{i}\right]=\frac{\alpha_{i}}{\alpha_{i}+\overline{\alpha}_{i}}=\frac{\alpha_{i}}{\sum_{c}\alpha_{c}}
\end{equation}
$$

The mode $\hat{\pi}=\arg\max_{\pi}p\left(\pi\right)$ of the Dirichlet distribution is given by:

$$
\begin{equation}
\hat{\pi}_{i}=\frac{\alpha_{i}-1}{\sum_{c}\alpha_{c}-C}
\end{equation}
$$

Using the above two properties, the MMSE and MAP estimates are easily defined under the posterior $p\left(\pi\vert y\right)$ we saw as:

$$
\begin{equation}
\hat{\pi}_{i}^{\text{MMSE}}=\frac{\alpha_{i}+N_{i}}{N+\sum_{c}\alpha_{c}}
\end{equation}
$$

 and:
 
$$
\begin{equation}
\hat{\pi}_{i}^{\text{MAP}}=\frac{\alpha_{i}+N_{i}-1}{N-C+\sum_{c}\alpha_{c}}
\end{equation}
$$

---

The posterior for the Dirichlet prior has a very natural interpretation as "pseudo-counts". 

If the Dirichlet prior is used to find the posterior in the equation above then we get:

$$
\begin{equation}
p\left(\pi|y\right)=\text{Dirichlet}\left(\pi|\;\alpha_{1}+N_{1},\cdots,\alpha_{C}+N_{C}\right)
\end{equation}
$$

When we scrutinize this it looks like the $\alpha_{i}$s are contributing to the posterior as additional "counts" to each of the classes; it seems as if the prior adds "counts" ahead of times to each $N_{c}$. Due to this behavior, many times in the literature the prior is said to add "pseudo-counts" to each of the classes. 

{% details Pseudo-counts in coin tosses%}
The simplest use-case of pseudo-counts is in coin tosses. Assume that we have a coin that comes up heads (represented by "0") with probability $\pi_{0}$ and tails (represented by "1") with probability $\pi_{1}$. Given a sequence of $N$ tosses $x_{1},\cdots,x_{N}$ with $x_{i}\in\\{ 0,1\\}$, the likelihood again takes the categorial form:

$$
\begin{equation}
p\left(x_{1},\cdots,x_{N}|\pi\right)=\pi_{0}^{N_{0}}\pi_{1}^{N_{1}}
\end{equation}
$$

where $N_{0}$ is the number of tosses which came up as heads and $N_{1}=N-N_{0}$ is the number of tosses that came up tails. 

In such an occasion, it again makes sense to use a Dirichlet prior over $\pi$. Let's assume:

$$
\begin{equation}
\pi\sim\text{Dirichlet}\left(\alpha_{0},\alpha_{1}\right)
\end{equation}
$$

Then, the posterior is given by:

$$
\begin{equation}
\pi|x_{1},\cdots,x_{N}\sim\text{Dirichlet}\left(\alpha_{0}+N_{0},\alpha_{1}+N_{1}\right)
\end{equation}
$$

The first interpretation that comes to mind of the prior is that before we saw the coin tosses $x_{1},\cdots,x_{N}$ we somehow observed an additional $\alpha_{0}+\alpha_{1}$ coin tosses, which we add to the sequence of coin tosses we observe.

Logically, this seems a bit weird. However, suppose the sequence we observed is 0000. In such a case, the ML estimate of $\pi_{0}$ will be:

$$
\begin{equation}
\hat{\pi}_{0}^{\text{ML}}=\frac{4}{4}=1\Leftrightarrow\hat{\pi}_{1}^{\text{ML}}=0
\end{equation}
$$

However, we might have cause to believe that the coin is much fairer than the ML estimate will lead us to believe - after all, getting heads 4 times in a row isn't so unbelievable, even with a fair coin. In such a case, using properties we will see below, if we use a prior we can get a more plausible estimate for $\pi_{1}$:

$$
\begin{equation}
\hat{\pi}_{1}^{\text{MMSE}}=\frac{\alpha_{1}}{4+\alpha_{0}+\alpha_{1}}>0
\end{equation}
$$

{% enddetails %}

<br>

# Discussion

Generative classification is, theoretically, a better approach than discriminative classification. If, and this is a big if, we have the true class-conditional distribution for the data $x$, then generative classification is provably optimal. However, in practice we never have the true class-conditional distribution.

Nonetheless, generative classification _is_ useful as it allows application of domain knowledge in a much simpler manner than discriminative classification. In this post, we only looked at modeling each class as a Gaussian, but there is no reason to confine ourselves to such a simple model for the data. Generative classification is intuitively very simple and can be applied with any class conditional prior.

In the next post we will look at using priors more complex then the simple Gaussians.


---
<span style="float:left"><a href="https://friedmanroy.github.io/BML/11_discriminative_classification/">← Discriminative classification</a></span><span style="float:right"><a href="https://friedmanroy.github.io/BML/13_gmms/">GMMs and EM →</a></span>