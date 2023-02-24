---
layout: distill
comments: true
title: Kernels

description: 

authors:
  - name: Roy Friedman

toc: 
  - name: Positive Semi-Definite Kernels
  - name: Constructing Kernels
  - name: RBF is a Valid Kernel
  - name: Connection to Linear Regression
---

<span style='float:left'><a href="https://friedmanroy.github.io/BML/7_evidence/">← Evidence Function</a></span><span style='float:right'><a href=""> →</a></span>
<br>
<br>

When we introduced linear regression, we allowed the usage of the
basis functions $h\left(\cdot\right)$ which mapped the inputs of
our problem to features in a different space, allowing the linear
regression to learn non-linear functions over the input space. This
greatly improves the expressiveness of the linear regression model,
but is still quite limited as we can only use a finite number of basis
functions. In this section we will introduce kernels, which will give
us much greater flexibility in the functions the linear regression
model can learn.

<br>
# Positive Semi-Definite Kernels

Before we begin to talk about kernels in earnest, let's start by defining what a kernel actually is. We will only be looking at positive semi-definite kernels in the course, which will allow us to use them in our setting (more on that later).

> **Definition: Positive Semi-Definite (PSD) Kernels** 
> A symmetric function $k:X\times X\rightarrow\mathbb{R}$ is called a PSD kernel on the set $X$ if the associated kernel matrix (also known as the Gram matrix) $K_{ij}=k\left(x_{i},x_{j}\right)$ is PSD for any set of distinct points $\left\{ x_{i}\right\} _{i=1}^{N}\subseteq X$

This definition alone already gives us some information on the type of functions that are valid kernels:

> **Claim:** As a by-product, if we have any two functions $g:\mathbb{R}^{m}\times\mathbb{R}^{m}\rightarrow\mathbb{R}$ and $h:\mathbb{R}^{m}\rightarrow\mathbb{R}^{n}$ such that: $g\left(x_{1},x_{2}\right)=h\left(x_{1}\right)^{T}h\left(x_{2}\right)$ then $g\left(\cdot,\cdot\right)$ is a PSD kernel (which we will just call a kernel for now). 

#### Proof:

Let's define:

$$
\begin{equation}
H^{T}=\left[\begin{matrix}-h\left(x_{1}\right)^{T}-\\
-h\left(x_{2}\right)^{T}-\\
\vdots\\
-h\left(x_{n}\right)^{T}-
\end{matrix}\right]\in\mathbb{R}^{m\times N}
\end{equation}
$$

where we have $N$ points $x_{1},x_{2},\ldots,x_{N}$. The Gram matrix is defined as:

$$
\begin{equation}
G=H^{T}H\Rightarrow G_{ij}=h\left(x_{i}\right)^{T}h\left(x_{j}\right)
\end{equation}
$$

As we have seen before, any the product of a matrix with it's transpose is a PSD matrix, which means that $G$ is PSD, which in turn means that $g\left(\cdot,\cdot\right)$ is a kernel. So, for any symmetric function $k\left(\cdot,\cdot\right)$, if we can show that it is the inner product of two vectors, then the function is a kernel. 
<span style='float:right'> $\square$ </span>

Now the connection to basis functions should be easy to see - any set of basis functions is part of a kernel.

---
#### Examples:

Let's go over some examples for valid and non-valid kernels. Hopefully this will build a good intuition of whether something is a PSD kernel or not.

-  $k\left(x,y\right)=c$ where $c>0$ is some constant
	This is quite obviously **a valid kernel**, since the Gram matrix will be made fully of $c$ s and we have:
	
	$$
\begin{align*}
a^{T}Ka & =a^{T}\left(c\boldsymbol{1}\right)\boldsymbol{1}^{T}a\\
 & =\sum_{i}a_{i}c\cdot\sum_{i}a_{i}\\
 & =c\sum_{i}a_{i}\sum_{i}a_{i}\\
 & =c\left(\sum_{i}a_{i}\right)^{2}\ge0
\end{align*}
$$


-  $k\left(x,y\right)=f\left(x\right)^{T}g\left(y\right)$ where $f\left(\cdot\right)\neq g\left(\cdot\right)$
	One of the basic definitions of a PSD kernel is that it is _symmetric_. However, for this function:
	
$$
k\left(x,y\right)=f\left(x\right)^{T}g\left(y\right)\neq f\left(y\right)^{T}g\left(x\right)=k\left(y,x\right)
$$
	so $k\left(\cdot,\cdot\right)$ **is not a valid kernel**


- $k\left(x,y\right)=\text{cov}[x,y]$
	When we build the Gram matrix using this function for any finite set of points, we actually build a covariance matrix, which we have shown is a PSD matrix. So this is **a valid kernel**


-  $k\left(x,y\right)=-x^{T}y$
	Recall that for a PSD matrix $A$ and any vector $v$ , the following must hold:
$$
v^{T}Av\ge0
$$
	Now, if we choose the unit vector $e_{i}$ as $v$ , we have:
$$
e_{i}^{T}Ae_{i}=A_{ii}
$$
	where $e_{i}=\left(0,\ldots,0,\overbrace{1}^{i\text{th index}},0,\ldots,0\right)$ . From this we directly see that for any $i$ , $A_{ii}\ge0$ must hold in order for $A$ to be PSD. In kernels this is equivalent to showing that $k\left(x,x\right)\ge0$ . In the case of this vector, $k\left(x,x\right)<0$ is a very real possibility, so there is no way that the Gram matrix will be PSD, so it is **not a valid kernel**

<br>
Okay, okay, that's enough examples. From the above there are two rules we can clearly see:

1. If there exists two vectors $x$ and $y$ such that $k\left(x,y\right)\neq k\left(y,x\right)$ then $k\left(\cdot,\cdot\right)$ is not a valid kernel
2. If there exists a vector $x$ such that $k\left(x,x\right)<0$ then $k\left(\cdot,\cdot\right)$ is not a valid kernel


However, there are some functions that do not violate these two rules but are still not kernels, so we need to be especially careful with how we define our functions and whether they are actually valid kernels or not.

---

# Constructing Kernels

We've shown that we can think of any set of basis functions as one part of a kernel, but if we always do this then we fall back to the problem of having to pick the set basis functions, instead of choosing the kernel directly. We have also shown that it is kind of a pain to prove if something is a valid kernel or not, in the general case. We will now go on to show how kernels can be constructed directly<d-footnote>The construction of kernels is the same as in Bishop 6.2, although slightly altered</d-footnote>, instead of going through basis functions.

Since a kernel is valid as long as the Gram matrix on any set of points is PSD, we can create building blocks, and stack them on top of each other to get more complex kernels. For any two valid kernels $k_{1}\left(\cdot,\cdot\right)$ and $k_{2}\left(\cdot,\cdot\right)$ the following functions will also be kernels:

1.  $c\cdot k_{1}\left(x,y\right)$ where $c>0$ is a constant
2.  $f\left(x\right)k_{1}\left(x,y\right)f\left(y\right)$ where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$
3.  $q\left(k_{1}\left(x,y\right)\right)$ where $q\left(\cdot\right)$ is any polynomial with non-negative coefficients
4.  $\exp\left(k_{1}\left(x,y\right)\right)$
5.  $k_{1}\left(x,y\right)+k_{2}\left(x,y\right)$
6.  $k_{1}\left(x,y\right)k_{2}\left(x,y\right)$
7.  $k\left(x,y\right)=x^{T}Ay$ where $A$ is PSD

Using these building blocks, we can create more and more complex kernels, as we see fit.

---

#### Example: Polynomial Kernel
Using the construction blocks, we saw that a polynomial of a kernel with non-negative coefficients is, itself, a valid kernel. Let's look at the specific case:

$$
\begin{equation}
k\left(x,y\right)=\left(x^{T}y\right)^{2}
\end{equation}
$$

where $x,y\in\mathbb{R}^{2}$ . This is clearly a kernel because claims (3) and (7) are used to construct it. However, we want to understand exactly why this is a valid kernel. This becomes quite clear when we take a closer look at the function:

$$
\begin{align*}
\left(x^{T}y\right)^{2} & =\left(x_{1}y_{1}+x_{2}y_{2}\right)^{2}\\
 & =x_{1}^{2}y_{1}^{2}+2x_{1}y_{1}x_{2}y_{2}+x_{2}^{2}y_{2}^{2}\\
 & =\left(\begin{matrix}x_{1}^{2} & \sqrt{2}x_{1}x_{2} & x_{2}^{2}\end{matrix}\right)\left(\begin{matrix}y_{1}^{2}\\
\sqrt{2}y_{1}y_{2}\\
y_{2}^{2}
\end{matrix}\right)\\
 & =p_{2}\left(x\right)^{T}p_{2}\left(y\right)
\end{align*}
$$

So, the kernel is actually the inner product between two basis functions, which in this case happen to be second order polynomials. The function $p_{2}\left(\cdot\right)$ maps a vector $x$ to all possible 2nd order terms, times some constants:

$$\left(\begin{matrix}x_{1}^{2} & \sqrt{2}x_{1}x_{2} & x_{2}^{2}\end{matrix}\right)^{T}$$ 
In the same manner, if we would have chosen a kernel of the form $\left(x^{T}y\right)^{m}$ , then it would map from the features into all possible $m$ -th order terms of the features. 

Now, instead, let's look at the following:

$$
\begin{align*}
\left(x^{T}y+1\right)^{2} & =\left(x_{1}y_{1}+x_{2}y_{2}+1\right)^{2}\\
 & =x_{1}^{2}y_{1}^{2}+x_{2}^{2}y_{2}^{2}+1+2x_{1}y_{1}+2x_{2}y_{2}+2x_{1}y_{1}x_{2}y_{2}
\end{align*}
$$

And we're not going to even bother writing this in vector form. Notice that now the expression holds all terms _up to_ the 2nd degree. This would also be true for any degree $m$ where $\left(x^{T}y+1\right)^{m}$ would expand all features up to (and including) the $m$ -th degree. It is also really easy to see that this is still a valid kernel, using the claims from earlier. The kernel:

$$
\begin{equation}
k_{m}\left(x,y\right)=\left(x^{T}y+1\right)^{m}
\end{equation}
$$

is called the polynomial kernel exactly because of the characteristics we described. 

---

# RBF is a Valid Kernel

A family of basis functions that are often used in practice are the radial basis functions (RBF). The most common function from this family is the Gaussian basis function:

$$
\begin{equation}
h_{\mu}\left(x\right)=\exp\left[-\frac{\left(x-\mu\right)^{2}}{2s^{2}}\right]
\end{equation}
$$

which we have already talked about. However, instead of defining the function's center ( $\mu$ ), we can define a kernel based on this basis function, that uses the distance between _any_ two points:

$$
\begin{equation}
k\left(x,y\right)=\exp\left[-\frac{1}{2}\left(x-y\right)^{T}A\left(x-y\right)\right]
\end{equation}
$$

where we have broadened our definition by adding $A$ as any PSD matrix. This kernel is very widely used in literature and in practice.

> **Claim:** The Gaussian kernel is a valid kernel

#### Proof:
For the proof we will use several of the claims for constructing kernels. We begin (as usual) by expanding the Mahalanobis distance:

$$
\begin{equation}
\left(x-y\right)^{T}A\left(x-y\right)=x^{T}Ax-2x^{T}Ay+y^{T}Ay
\end{equation}
$$

We can now rewrite the function as:

$$
\begin{align}
k\left(x,y\right) & =\exp\left[-\frac{1}{2}x^{T}Ax\right]\exp\left[x^{T}Ay\right]\exp\left[-\frac{1}{2}y^{T}Ay\right]\\
&\stackrel{\Delta}{=}f\left(x\right)\exp\left[\underbrace{x^{T}Ay}_{q\left(x,y\right)}\right]f\left(y\right)
\end{align}
$$

Notice how we brought the form of the whole function as the form in claim (2) (i.e., multiplied by a function on either side). Since $A$ is a PSD matrix, $q\left(x,y\right)$ must also be a valid kernel. Using claim (4), the exponent of a kernel is also a kernel, therefore $k\left(\cdot,\cdot\right)$ is a valid kernel.
<span style='float:right'> $\square$ </span>

<br>
The Gaussian kernel can actually be rewritten as:

$$
\begin{align}
k\left(x,y\right) & =f\left(x\right)\left(\sum_{n=0}^{\infty}\frac{1}{n!}\left(x^{T}Ay\right)^{n}\right)f\left(y\right)\nonumber \\
 & =f\left(x\right)\left(\sum_{n=0}^{\infty}\frac{1}{n!}k_{n}\left(x,Ay\right)\right)f\left(y\right)
\end{align}
$$

where $k_{n}\left(\cdot,\cdot\right)$ is the $n$ -th order polynomial kernel. Remember how we opened up the polynomial kernel into the inner product between two vectors? For this kernel, we would have to build infinitely long vectors, with every polynomial degree in it, and then find the inner product of these two vectors. In this sense, we can think of the kernel as a basis function $h\left(\cdot\right)$ that maps inputs into infinite feature spaces.

<br>
# Connection to Linear Regression

The above is well and good, but it is not readily apparent how all
of this is connected to what we talked about up to this point. We
will now give an intuition for how this is connected to ridge regression,
which can be viewed as a private case of Bayesian linear regression<d-footnote>Bishop 6.1</d-footnote>.

First, remember that the loss function for ridge regression is given by:

$$
\begin{equation}
L\left(\theta\right)=\|H\theta-y\|^{2}+\lambda\|\theta\|^{2}
\end{equation}
$$

Deriving this by $\theta$ and equating to 0 we see that:

$$
\begin{equation}
2\frac{\partial}{\partial\theta}L\left(\theta\right)=-H^{T}H\theta+H^{T}y+\lambda\theta\stackrel{!}{=}0
\end{equation}
$$

We can now define:

$$
\begin{align}
\theta & =\frac{1}{\lambda}H^{T}\left(H\theta-y\right)=\frac{1}{\lambda}\sum_{i}\overbrace{\left(h\left(x_{i}\right)^{T}\theta-y_{i}\right)}^{\alpha_{i}}h\left(x_{i}\right)\\
 & \stackrel{\Delta}{=}\sum_{i}\alpha_{i}h\left(x_{i}\right)\stackrel{\Delta}{=} H^{T}\alpha
\end{align}
$$

So we can rewrite the MLE solution for $\theta$ as a linear function of the basis functions in $H$ , where $\alpha_{i}$ is the coefficient for the basis functions over the $i$ -th sample $h\left(x_{i}\right)$ .

The minimum of $L$ with respect to $\alpha$ is:

$$
\begin{align*}
L\left(\alpha\right) & =\|HH^{T}\alpha-y\|^{2}+\lambda\|H^{T}\alpha\|^{2}\\
\Rightarrow\frac{\partial}{\partial\alpha}L\left(\alpha\right) & =HH^{T}HH^{T}\alpha-HH^{T}y+\lambda HH^{T}\alpha\stackrel{!}{=}0
\end{align*}
$$

If we assume that $HH^{T}$ is invertible, we get:

$$
\begin{equation}
\hat{\alpha}_{ML}=\left(HH^{T}+I\lambda\right)^{-1}y
\end{equation}
$$

which is very similar to what we saw for the MLE for $\theta$ . Now, let's plug this into the linear regression function:

$$
\begin{align}
f\left(x\right) & =\hat{\theta}_{ML}^{T}h\left(x\right)=\hat{\alpha}_{ML}^{T}Hh\left(x\right)\\
 & =y^{T}\left(HH^{T}+I\lambda\right)^{-1}Hh\left(x\right)
\end{align}
$$

Defining the Gram matrix $K=HH^{T}$ such that $K_{ij}=h\left(x_{i}\right)^{T}h\left(x_{j}\right)$ and the vector $k\left(x\right)$ where $k_{i}\left(x\right)=h\left(x\right)^{T}h\left(x_{i}\right)$ , then we can rewrite the above as:

$$
\begin{equation}
f\left(x\right)=y^{T}\left(K+I\lambda\right)^{-1}k\left(x\right)
\end{equation}
$$

Suppose that, instead of defining the basis functions $h\left(\cdot\right)$ , we define a kernel $k\left(x_{i},x_{j}\right)$ . While before it was not so obvious how this would fit into the structure of the linear regression, now all we need to define is $K_{ij}=k\left(x_{i},x_{j}\right)$ and $k_{i}\left(x\right)=k\left(x,x_{i}\right)$ in order to get the prediction for a new point $x$ .

Now the reason we allow PSD and not only PD kernels might be easier to see. Since the matrix we are inverting is $\left(K+I\lambda\right)$ , even if the Gram matrix $K$ is not PD (i.e. not invertible) and only PSD, because of the added term $I\lambda$ , the whole matrix is PD. Under this reasoning, and because forming PSD kernels is easier than forming PD kernels, we can relax our restrictions on the form of the kernels and use PSD kernels, giving us just a bit more freedom.

<br>

---
<span style='float:left'><a href="https://friedmanroy.github.io/BML/7_evidence/">← Evidence Function</a></span><span style='float:right'><a href=""> →</a></span>
