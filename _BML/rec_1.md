---
layout: distill
title: Gaussian Distribution
description: The Gaussian distribution is hands-down the most-used distribution in machine learning. This post will go through key aspects of the normal distribution and its representations.

authors:
  - name: Roy Friedman
    affiliations:
      name: Hebrew University

toc:
  - name: Definition
  - name: Geometry of the Gaussian Distribution
  - name: The Derivative Trick
  - name: Completing the Squares
  - name: Extras

place: 2
---

The distribution that is seen most often in ML (and statistics) is the Gaussian distribution, also called the _normal distribution_. The reason this distribution is so commonly used is because of two reasons: it is empirically observed in the wild many times and, perhaps more importantly, it is mathematically very simple to use the Gaussian distribution (we will see exactly how later on). This post will delve into the definition and properties of the Gaussian distribution[^1].

# Definition

A random variable $x$ is said to have a Gaussian distribution if it's PDF has the following form:

$$
\begin{equation}
p\left(x\right)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left[-\frac{1}{2\sigma^{2}}\left(x-\mu\right)^{2}\right]
\end{equation}
$$
Notice that this distribution can be completely described by the two parameters $\mu$ and $\sigma$ - the mean and variance of the Gaussian. Because of this, we will usually write:

$$
\begin{equation}
x\sim\mathcal{N}\left(\mu,\sigma^{2}\right)
\end{equation}
$$
to indicate that $x$ is a Gaussian random variable, with the two parameters $\mu$ and $\sigma$ . In the same manner, we denote the PDF by:

$$
\begin{equation}
p\left(x\right)=\mathcal{N}\left(x\,\mid \,\mu,\sigma^{2}\right)\qquad{\scriptstyle \left(\equiv\mathcal{N}\left(x\,;\,\mu,\sigma^{2}\right)\right)}
\end{equation}
$$

The conditioning sign (or semi-colon) in $\mathcal{N}\left(x\,\mid \,\mu,\sigma^{2}\right)$ is to show that $x$ is the variable that we are interested in, while $\mu$ and $\sigma$ are the parameters that define the distribution (so, given a $\mu$ and a $\sigma$, we know the PDF of $x$ ). 


<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_1/1D_vis.png"  
alt="Visualization of a 1D Gaussian"  
style="display: inline-block; margin: 0 auto; ">
</p>
<div class="caption">
    Figure 1: examples of 1D Gaussian distributions with different means ( $\mu$ ) and standard deviations ( $\sigma$ ).
</div>

---

The multivariate version for a $d$ -dimensional random vector $x\in\mathbb{R}^{d}$
is defined as:

$$
\begin{equation}
\mathcal{N}\left(x\,\mid \,\mu,\Sigma\right)=\frac{1}{\sqrt{\left(2\pi\right)^{d}\mid \Sigma\mid }}\exp\left[-\frac{1}{2}\left(x-\mu\right)^{T}\Sigma^{-1}\left(x-\mu\right)\right]
\end{equation}
$$

and in this case $\mu$ is also a vector and $\Sigma$ is a symmetrical $n\times n$ matrix. The term $D_{M}\left(x\,\mid \,\mu,\Sigma\right)^{2}=\left(x-\mu\right)^{T}\Sigma^{-1}\left(x-\mu\right)$ is often called the _Mahalanobis distance_ and is denoted with $\Delta$. The multivariate version for the Gaussian distribution is also called
the _multivariate normal_ (MVN) distribution.


<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_1/2D_vis.png"  
alt="Visualization of a 2D Gaussian"  
style="display: inline-block; margin: 0 auto;">
</p>
<div class="caption">
    Figure 2: example of a 2D Gaussian distribution. On the left is the heatmap of the distribution - darker means higher density. On the right is the contour at $\Delta=1$ overlayed on top of samples from the distribution. The contours of the distribution are ellipses aligned and scaled according to the eigenvectors and eigenvalues of the covariance matrix.
</div>


# Geometry of the Gaussian Distribution

In 1D, the Gaussian distribution takes the form of the famous bell curve in figure 1 and is easy to view. However, in multiple dimensions it is not so clear what the geometry of the distribution actually looks like. We can gain insight by considering the EVD of the covariance matrix $\Sigma$ (remember, this decomposition exists since $\Sigma$ is symmetric):

$$
\Sigma=UDU^{T}
$$
where $D$ is a diagonal matrix with the eigenvalues $\lambda_{i}$ on the diagonal and $U$ is an orthonormal matrix (so $UU^{T}=I$ ) with the eigenvectors $u_{i}$ as it's rows, such that for all $i$ :

$$
\begin{equation}
\Sigma u_{i}=\lambda_{i}u_{i}
\end{equation}
$$
Recall that the eigenvectors are orthogonal to each other, and we can choose eigenvectors that are normalized, so for all $i\neq j$ we have $u_{i}^{T}u_{j}=0$ and $u_{i}^{T}u_{i}=1$ . 

We can rewrite this decomposition (using the basis defined by the eigenvectors) as:
$$
\begin{align}
\Sigma & =\sum_{i}\lambda_{i}u_{i}u_{i}^{T}
\end{align}
$$
The inverse of this matrix is then given by:
$$
\begin{equation}
\Sigma^{-1}=\sum_{i}\frac{1}{\lambda_{i}}u_{i}u_{i}^{T}
\end{equation}
$$
This allows us to rewrite the Mahalanobis distance as follows:
$$
\begin{align}
\label{eq:gauss-ellipse}
\Delta & \equiv D_{M}\left(x\,\mid \,\mu,\Sigma\right)^{2}\\
 & =\sum_{i}\frac{1}{\lambda_{i}}\left(x-\mu\right)^{T}u_{i}u_{i}^{T}\left(x-\mu\right)\\
 & =\sum_{i}\frac{\left(u_{i}^{T}\left(x-\mu\right)\right)^{2}}{\lambda_{i}}\equiv\sum_{i}\frac{y_{i}^{2}}{\lambda_{i}}
\end{align}
$$
where we defined $y_{i}=u_{i}^{T}\left(x-\mu\right)$ . Notice that the density will be constant on the surfaces where $\Delta$ is constant. The shape described by \eqref{eq:gauss-ellipse} is an _ellipse_ with radii equal to $\lambda_{i}^{1/2}$ , centered around $\mu$ .


This is really clear in the 2D case:
$$
\begin{equation}
\Delta=\left(\frac{u_{1}^{T}\left(x-\mu\right)}{\sqrt{\lambda_{1}}}\right)^{2}+\left(\frac{u_{2}^{T}\left(x-\mu\right)}{\sqrt{\lambda_{2}}}\right)^{2}
\end{equation}
$$
So in 2D, all of the _contour lines_ (which are lines that have the same density along the PDF) will always be ellipses; in the multivariate case they will be ellipsoids (which is an ellipse in more dimensions, kind of). Figure 2 (right) shows this explicitly - the Gaussian is centered around the mean $\mu$ , the contour of $\Delta=1$ is an ellipse with axes aligned and scaled by the eigenvectors and square root of the eigenvalues of the covariance matrix.


# The Derivative Trick

The Gaussian distribution is, _by definition_, any distribution that is the exponent of a quadratic function, i.e. any distribution of the form:
$$
\begin{equation}
p\left(x\right)\propto\exp\left[-x^{T}\Gamma x+b^{T}x+c\right]
\end{equation}
$$
is Gaussian (even though it doesn't seem like it at first). In this course, we will see distributions with a form similar to the above, but will want to find the actual parameters ( $\mu$ and $\Sigma$ ) that define the Gaussian, instead of leaving it as it is written above.


However, we can go even further:
$$
\begin{align}
\Delta & =\frac{1}{2}\left(x-\mu\right)^{T}\Sigma^{-1}\left(x-\mu\right)\nonumber \\
\Leftrightarrow\frac{\partial\Delta}{\partial x} & =\Sigma^{-1}\left(x-\mu\right)\\
\frac{\partial^{2}\Delta}{\partial x\partial x^{T}} & =\Sigma^{-1}
\end{align}
$$
So, if we know that $p\left(x\right)$ is a Gaussian, and we want to find $\mu$ and $\Sigma$ , we can simply differentiate $-\log p\left(x\right)$ and try to manipulate the resulting expression until we get:
$$
\begin{equation}
\frac{\partial}{\partial x}\left(-\log p\left(x\right)\right)=\Sigma^{-1}\left(x-\mu\right)
\end{equation}
$$

---

## Conditional Distribution of a Gaussian

An important property of the multivariate Gaussian distribution is that if two sets of variables are jointly Gaussian, then the conditional distribution of one set on the other is also Gaussian.

Consider a Gaussian variable separated into 2 parts:
$$
\begin{equation}
x=\left(\begin{matrix}x_{a}\\
x_{b}
\end{matrix}\right)
\end{equation}
$$
such that $x\sim\mathcal{N}\left(\mu,\Sigma\right)$ . We can divide the mean and the covariance in a fitting manner:
$$
\begin{equation}
\mu=\left(\begin{matrix}\mu_{a}\\
\mu_{b}
\end{matrix}\right),\quad\Sigma=\left(\begin{matrix}\Sigma_{aa} & \Sigma_{ab}\\
\Sigma_{ba} & \Sigma_{bb}
\end{matrix}\right)
\end{equation}
$$
Since the covariance is symmetrical, we know that $\Sigma_{ab}=\Sigma_{ba}^{T}$ and that $\Sigma_{aa}$ and $\Sigma_{bb}$ are symmetrical. Actually, it will be easier to use the precision matrix $\Lambda=\Sigma^{-1}$ and divide it up in the same manner:
$$
\begin{equation}
\Lambda=\left(\begin{matrix}\Lambda_{aa} & \Lambda_{ab}\\
\Lambda_{ba} & \Lambda_{bb}
\end{matrix}\right)
\end{equation}
$$
Note that $\Lambda_{aa}\neq\Sigma_{aa}^{-1}$ ! Later we will find out how each part of $\Lambda$ relates to each part of $\Sigma$ .

Let's start by finding an expression for the conditional distribution $p\left(x_{a}\mid x_{b}\right)$ . We can find this distribution by evaluating the distribution of $p\left(x_{a},x_{b}\right)$ while fixing $x_{b}$ to a certain value and re-normalizing (the conditional distribution is a legal distribution). We will start by rewriting the quadratic term and it's parts:
$$
\begin{align}
-\frac{1}{2}\left(x-\mu\right)^{T}\Lambda\left(x-\mu\right)=-\frac{1}{2}\left[\left(x_{a}-\mu_{a}\right)^{T}\Lambda_{aa}\left(x_{a}-\mu_{a}\right)\right. & +\left(x_{a}-\mu_{a}\right)^{T}\Lambda_{ab}\left(x_{b}-\mu_{b}\right)\nonumber \\
+\left(x_{b}-\mu_{b}\right)^{T}\Lambda_{ba}\left(x_{a}-\mu_{a}\right) & \left.+\left(x_{b}-\mu_{b}\right)^{T}\Lambda_{bb}\left(x_{b}-\mu_{b}\right)\right]
\end{align}
$$
This is still a quadratic expression w.r.t. $x_{a}$ , so the conditional distribution $p\left(x_{a}\mid x_{b}\right)$ will also be Gaussian. Because the form of the Gaussian is not very flexible, as long as we find what the quadratic term is equal (the one in the exponent), the normalization will work itself out (since the conditional is also a distribution that must integrate up to 1). 

We can now use the derivative trick! Defining:
$$
\begin{equation}
\Delta=\frac{1}{2}\left(x-\mu\right)^{T}\Sigma^{-1}\left(x-\mu\right)
\end{equation}
$$
Starting with the first derivative:
$$
\begin{equation}\label{eq:mean-form-1}
\frac{\partial\Delta}{\partial x_{a}}=\Lambda_{aa}\left(x_{a}-\mu_{a}\right)+\Lambda_{ab}\left(x_{b}-\mu_{b}\right)
\end{equation}
$$
The second derivative will give us:
$$
\begin{equation}
\begin{split}\frac{\partial^{2}\Delta}{\partial x_{a}\partial x_{a}^{T}} & =\Lambda_{aa}\end{split}
\end{equation}
$$
So we know that the covariance is equal to $\Sigma_{a\mid b}=\Lambda_{aa}^{-1}$ . Using this new found knowledge, we can find the mean, if we can rewrite equation \eqref{eq:mean-form-1} as $\Lambda_{aa}\left(x_{a}-\mu_{a\mid b}\right)$ for some $\mu_{a\mid b}$ . Let's try to do this. Recall that $\Lambda_{aa}$ is invertible, so we can write:
$$
\begin{align}
\Lambda_{aa}\left(x_{a}-\mu_{a}\right)+\Lambda_{ab}\left(x_{b}-\mu_{b}\right) & =\Lambda_{aa}\left(x_{a}-\mu_{a}+\Lambda_{aa}^{-1}\Lambda_{ab}\left(x_{b}-\mu_{b}\right)\right)\nonumber \\
 & =\Lambda_{aa}\left[x_{a}-\left(\mu_{a}-\Lambda_{aa}^{-1}\Lambda_{ab}\left(x_{b}-\mu_{b}\right)\right)\right]\nonumber \\
 & \stackrel{\Delta}{=}\Lambda_{aa}\left(x_{a}-\mu_{a\mid b}\right)
\end{align}
$$
Which means that the conditional distribution is parameterized by:
$$
\begin{align}
\mu_{a\mid b} & =\mu_{a}-\Lambda_{aa}^{-1}\Lambda_{ab}\left(x_{b}-\mu_{b}\right)\\
\Sigma_{a\mid b} & =\Lambda_{aa}^{-1}
\end{align}
$$

Okay, now all that remains is to find what $\Lambda_{aa}$ and $\Lambda_{ab}$ are equal to in terms of $\Sigma$ . To do this, we will use the following identity for partitioned matrices:
$$
\begin{equation}
\left(\begin{array}{cc}
A & B\\
C & D
\end{array}\right)^{-1}=\left(\begin{array}{cc}
M & -MBD^{-1}\\
-D^{-1}CM & \quad D^{-1}+D^{-1}CMBD^{-1}
\end{array}\right)^{-1}
\end{equation}
$$
where $M=\left(A-BD^{-1}C\right)^{-1}$ is called the _Schur component_ of the matrix with respect to the sub-matrix $D$ . Using our earlier partitioning, we get:
$$
\begin{align}
\Lambda_{aa} & =\left(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\right)^{-1}\\
\Lambda_{ab} & =-\Lambda_{aa}\Sigma_{ab}\Sigma_{bb}^{-1}
\end{align}
$$
Finally, we have the expressions needed to describe the conditional distribution:
$$
\begin{align}
p\left(x_{a}\mid x_{b}\right) & =\mathcal{N}\left(x_{a}\,\mid \,\mu_{a\mid b},\Sigma_{a\mid b}\right)\\
\mu_{a\mid b} & =\mu_{a}+\Sigma_{ab}\Sigma_{bb}^{-1}\left(x_{b}-\mu_{b}\right)\nonumber \\
\Sigma_{a\mid b} & =\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\nonumber 
\end{align}
$$
Note that in this case, the conditional distribution is much easier to describe in terms of the precision matrix instead of the covariance matrix. 

When implementing the code for this, it may be simpler to save the precision matrix (as well as the covariance matrix) in memory to easily compute the conditional distribution.


<p align="center">
<img  
src="https://friedmanroy.github.io/assets/bml_figs/rec_1/cond_vis.png"  
alt="Visualization of a 1D Gaussian"  
style="display: inline-block; margin: 0 auto; ">
</p>
<div class="caption">
    Figure 3: visualization of the conditional of a bivariate Gaussian; plots of $p\left(x_{a}\mid x_{b}\right)$ for various values of $x_{b}$ . Notice how the variance doesn't change for different values of $x_{b}$ , only the mean of the conditional.
</div>


# Completing the Squares

While the derivative trick is very useful, we can't always use it, since we might lose information that we want to keep when differentiating. In such cases, we can use a different trick - completing the squares. 

"Completing the squares" means that we want to turn a quadratic _function_ into the quadratic _form_ (plus some residuals). Suppose we have the following expression:
$$
\begin{equation}
f\left(x\right)=x^{T}Ax+2x^{T}b+c
\end{equation}
$$
In this case, completing the squares means we would like to bring $f\left(x\right)$ to the form:
$$
\begin{equation}
f\left(x\right)=\underbrace{\left(x+\boxed{?}\right)^{T}\boxed{\boxed{?}}\left(x+\boxed{?}\right)}_{\text{depends on }x}+\underbrace{g\left(A,b,c\right)}_{\text{const w.r.t }x}
\end{equation}
$$
where $\boxed{?}$ stands in for some vector and $\boxed{\boxed{?}}$ stands in for some matrix. For the case presented above, we can do so in the following manner (assuming $A$ is invertible[^2]):
$$
\begin{align}
\label{eq:quad-full}
f\left(x\right) & =x^{T}Ax+2x^{T}b+c\\
 & =x^{T}Ax+2x^{T}AA^{-1}b+c\\
 & =x^{T}Ax+2x^{T}AA^{-1}b-b^{T}A^{-1}AA^{-1}b+b^{T}A^{-1}AA^{-1}b+c\\
 & =\left(x+A^{-1}b\right)^{T}A\left(x+A^{-1}b\right)-\underbrace{b^{T}A^{-1}b+c}_{\text{const w.r.t }x}
\end{align}
$$

Having written down the full quadratic form in \eqref{eq:quad-full}, we can now understand which terms we lose when we use the derivative trick. When we differentiate $f\left(\cdot\right)$ with respect to $x$ , we willingly drop all of the terms that are constant with respect to $x$ - in this case, we would lose all information regarding $g\left(A,b,c\right)$ . 

For example, suppose we want to find:
$$
\begin{equation}
p\left(y\right)\propto\intop\exp\left[-\frac{1}{2}\left(x^{T}\Gamma x+2x^{T}h\left(y\right)\right)\right]dx
\end{equation}
$$
If we use the derivative trick to find the form of the Gaussian in the exponent, we would lose all information regarding $y$ ! This information is obviously important - we want to find $p\left(y\right)$ , after all. Instead, plugging into the formula from \eqref{eq:quad-full}, we have:
$$
\begin{align}
\intop\exp\left[-\frac{1}{2}\left(x^{T}\Gamma x+2x^{T}h\left(y\right)\right)\right]dx & =\intop\exp\left[-\frac{1}{2}\left(x-\Gamma^{-1}h\left(y\right)\right)^{T}\Gamma\left(x-\Gamma^{-1}h\left(y\right)\right)+\frac{1}{2}h\left(y\right)^{T}\Gamma^{-1}h\left(y\right)\right]dx\\
 & \propto e^{\frac{1}{2}h\left(y\right)^{T}\Gamma^{-1}h\left(y\right)}\intop\mathcal{N}\left(x\,\mid \,\Gamma^{-1}h\left(y\right),\Gamma^{-1}\right)dx\\
 & =e^{\frac{1}{2}h\left(y\right)^{T}\Gamma^{-1}h\left(y\right)}\propto p\left(y\right)
\end{align}
$$

---

## Marginal Distribution of a Gaussian

Another important property of the Gaussian distribution is that its marginals are also Gaussian, which is what we will show here.

Again, we consider a Gaussian variable separated into 2 parts:
$$
\left(\begin{matrix}x_{a}\\
x_{b}
\end{matrix}\right)\sim\mathcal{N}\left(\left(\begin{matrix}\mu_{a}\\
\mu_{b}
\end{matrix}\right),\left(\begin{matrix}\Sigma_{aa} & \Sigma_{ab}\\
\Sigma_{ba} & \Sigma_{bb}
\end{matrix}\right)\right)
$$
and we will again define $\Lambda=\Sigma^{-1}$ . We want to find $p\left(x_{a}\right)$ .


Our battle plan is to first find all of the dependence on $x_{b}$ ,
and to integrate it out. If we can do this without losing track of
$x_{a}$ , we will be able to find the marginal distribution and win.
Notice that the Mahalanobis distance is quadratic in $x_{b}$ (as
usual), so the final form we will get to will be something like:

$$
p\left(x_{a}\right)=f\left(x_{a}\right)\intop\mathcal{N}\left(x_{b}\,\mid \,\mu_{x_{b}},\Sigma_{x_{b}}\right)dx_{b}=f\left(x_{a}\right)
$$

So, let's try to open up the Mahalanobis distance (the part in the
exponent) and separate it into two groups: terms that contain $x_{b}$
and those that don't. 


We begin by defining $y=x-\mu$ , which in this case will allow us
to write the Mahalanobis distance as:
$$
\begin{align}
\Delta & =\frac{1}{2}\left(\begin{matrix}y_{a}\\
y_{b}
\end{matrix}\right)^{T}\left(\begin{matrix}\Lambda_{a} & B\\
B^{T} & \Lambda_{b}
\end{matrix}\right)\left(\begin{matrix}y_{a}\\
y_{b}
\end{matrix}\right)\nonumber \\
 & =\frac{1}{2}\left(\begin{matrix}y_{a}\\
y_{b}
\end{matrix}\right)^{T}\left(\begin{matrix}\Lambda_{a}y_{a}+By_{b}\\
B^{T}y_{a}+\Lambda_{b}y_{b}
\end{matrix}\right)\nonumber \\
 & =\frac{1}{2}y_{a}^{T}\Lambda_{a}y_{a}+\underbrace{\frac{1}{2}\left[2y_{b}^{T}By_{a}+y_{b}^{T}\Lambda_{b}y_{b}\right]}_{\left(*\right)}
\end{align}
$$
Note that if we find the marginal of $y_{a}$ , we effectively find the marginal of $x_{a}$ , only we don't have to keep track of $\mu_{a}$ and $\mu_{b}$ ! We now need to complete the squares in $\left(*\right)$ to find the complete dependence on $y_{b}$ :
$$
\begin{align}
2y_{b}^{T}By_{a}+y_{b}^{T}\Lambda_{b}y_{b} & =y_{b}^{T}\Lambda_{b}y_{b}+2y_{b}^{T}By_{a}\nonumber \\
 & =y_{b}^{T}\Lambda_{b}y_{b}+2y_{b}^{T}\Lambda_{b}\Lambda_{b}^{-1}By_{a}\nonumber \\
 & =y_{b}^{T}\Lambda_{b}y_{b}+2y_{b}^{T}\Lambda_{b}\Lambda_{b}^{-1}By_{a}+y_{a}^{T}B^{T}\Lambda_{b}^{-1}\Lambda_{b}\Lambda_{b}^{-1}By_{a}-y_{a}^{T}B^{T}\Lambda_{b}^{-1}\Lambda_{b}\Lambda_{b}^{-1}By_{a}\nonumber \\
 & =\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)^{T}\Lambda_{b}\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)-y_{a}^{T}B^{T}\Lambda_{b}^{-1}By_{a}
\end{align}
$$

We add this back to the full Mahalanobis distance to get:
$$
\begin{equation}
\Delta=\frac{1}{2}y_{a}^{T}\left(\Lambda_{a}-B^{T}\Lambda_{b}^{-1}B\right)y_{a}+\frac{1}{2}\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)^{T}\Lambda_{b}\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)
\end{equation}
$$

So our distribution is:

$$
\begin{align}
p\left(y_{a}\right) & \propto\exp\left[-\frac{1}{2}y_{a}^{T}\left(\Lambda_{a}-B^{T}\Lambda_{b}^{-1}B\right)y_{a}\right]\intop\exp\left[-\frac{1}{2}\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)^{T}\Lambda_{b}\left(y_{b}+\Lambda_{b}^{-1}By_{a}\right)\right]dy_{b}\nonumber \\
 & \propto\exp\left[-\frac{1}{2}y_{a}^{T}\left(\Lambda_{a}-B^{T}\Lambda_{b}^{-1}B\right)y_{a}\right]
\end{align}
$$

which is definitely Gaussian! We were allowed to do the second move because the integral will be the normalization term of the Gaussian, which is a function of $\Lambda_{b}$ - which is constant with respect to $y_{a}$ (and so is eaten up by the $\propto$ sign).

Finally, we see that:
$$
\begin{equation}
y_{a}\sim\mathcal{N}\left(0,\left(\Lambda_{a}-B^{T}\Lambda_{b}^{-1}B\right)^{-1}\right)\Rightarrow x_{a}\sim\mathcal{N}\left(\mu_{a},\left(\Lambda_{a}-B^{T}\Lambda_{b}^{-1}B\right)^{-1}\right)
\end{equation}
$$

and all that remains is to find what the covariance is equal to in terms of $\Sigma$ . To do this, we use the same identity we saw in the previous example:
$$
\begin{equation}
\Sigma_{aa}=\left(\Lambda_{aa}-B^{T}\Lambda_{b}^{-1}B\right)^{-1}
\end{equation}
$$

So, the marginal is:
$$
\begin{equation}
x_{a}\sim\mathcal{N}\left(\mu_{a},\Sigma_{aa}\right)
\end{equation}
$$
which really makes you wonder why we did all of that hard work.

# Extras 

We saw the so called "derivative trick" and how completing the squares can also be of help, but it might not be obvious when to use each approach. First, remember that whenever we see a distribution of the form:
$$
\begin{equation}
p\left(x,y\right)\propto\exp\left[-x^{T}\Gamma x+b\left(y\right)^{T}x+g\left(y\right)\right]
\end{equation}
$$

then $p\left(x\right)$ and $p\left(x\mid y\right)$ will be Gaussians[^3], and we will usually want to find the "Gaussian form" we know and love. Once we figured that out, we can try to ask "how can we find the Gaussian form?", and the answer will usually be one of the following methods:

* If we don't care about $p\left(y\right)$ or $p\left(y\mid x\right)$ at all, i.e. we want to specifically find $p\left(x\right)$ or $p\left(x\mid y\right)$ , then we can use the derivative trick
* If we need to know $p\left(y\right)$ or $p\left(y\mid x\right)$ explicitly as well as $p\left(x\right)$ or $p\left(x\mid y\right)$ , then completing the squares is usually the easiest way
* When all else fails, but we know that what we are looking for is Gaussian, we can calculate the expectations and covariance explicitly, since a Gaussian is completely defined by these two values

Once you fully understand why each method works, it will become quite clear when you should use each of them.

---


[^1]: See [Bishop 2.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) for a _much_ more extensive introduction to the Gaussian distribution

[^2]: We can also do this when $A$ is not invertible, in which case we will need to use the pseudo-inverse of $A$ such that $AA^{\dagger}=I$ . 

[^3]: This is a slightly more general statement than what we showed here, but you can verify the validity for yourself in these cases as well