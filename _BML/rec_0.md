---
layout: distill
title: Linear Algebra and Probability
description: All of the material in linear algebra and probability needed to understand Bayesian machine learning.

toc:
  - name: Linear Algebra
  - name: Probability
---

# Linear Algebra

Almost all of the relevant material in Bayesian machine learning happens in a high dimensional space. In order to be comfortable with operations in many dimensions, it's crucial to have a good understanding of linear algebra and the notation conventions. This is the purpose of this section.

## Vectors

A vector is a list of numbers which define a point in space:
$$
x\in\mathbb{R}^{d}\qquad x=\left[x_{1},x_{2},...,x_{d}\right]^{T}
$$

Here the $\mathbb{R}$ indicates that all of the elements in the vector are real and the superscript $\mathbb{R}^d$ tells us that there are $d$ numbers in the vector. The inner product of two vectors is defined as:
$$
\begin{equation}
\left\langle x,y\right\rangle \equiv x^{T}y\stackrel{\Delta}{=}\sum_{i=1}^{d}x_{i}y_{i}
\end{equation}
$$

(the sign "$\equiv$ " is to show that the first way of writing the inner product and the second one mean the same thing in this context). For any three (real) vectors $x,y,z\in\mathbb{R}^d$ and a scalar $a\in\mathbb{R}$, the inner product has the following properties:

* Linearity: $\left\langle ax+z,\ y\right\rangle \equiv \left(ax+z\right)^Ty=a(x^Ty)+z^Ty$
* Symmetry: $x^Ty=y^Tx$
* Positive-definite: $x^Tx\ge0$ and $x^Tx=0\Leftrightarrow x=0$

Where $x=0$ means that $\forall i\in [d]\ \ x_i=0$. Geometrically, the inner product is the projection of one vector onto another, which will be a very useful intuition to keep in mind and raises another important definition. Two vectors $x,y\in\mathbb{R}^d$ such that:
$$
\begin{equation}
x^Ty=0
\end{equation}
$$
are said to be orthogonal.

The inner product is just one way of multiplying the vectors and in this course we will also use the _outer product_ of 2 vectors:
$$
\begin{equation}

x\in\mathbb{R}^{d},y\in\mathbb{R}^{m}\quad xy^{T}=\left[\begin{array}{cccc}

x_{1}y_{1} & x_{1}y_{2} & \cdots & x_{1}y_{m}\\

x_{2}y_{1} & x_{2}y_{2} & \cdots & x_{2}y_{m}\\

\vdots & & & \vdots\\

x_{d}y_{1} & x_{d}y_{2} & \cdots & x_{d}y_{m}

\end{array}\right]\in\mathbb{R}^{d\times m}

\end{equation}
$$
As you can see, the outcome of an outer product is a matrix, instead of a scalar.

Another important quality of vectors are _norms_, which are metrics for their distance from the origin. The most commonly used norm is the Euclidean norm, also called the $\ell_2$ norm, defined as:
$$
\begin{equation}

\|x\|_{2}\stackrel{\Delta}{=}\sqrt{x^{T}x}=\sqrt{\sum_{i}x_{i}^{2}}

\end{equation}
$$
Sometimes we will write the Euclidean distance simply as $\|\cdot\|$ instead of adding the subscript "2" - this is because it is by far the most common norm we will see and is usually much simpler to use than other norms. Another norm that is used quite often is the $\ell_1$ norm, defined as:
$$
\begin{equation}

\|x\|_{1}\stackrel{\Delta}{=}\sum_{i}\left|x_{i}\right|

\end{equation}
$$

Apart from just measuring the distance from the origin, norms also allow us to measure the distance from other vectors. The $\ell_2$ distance between $x$ and $y$ is defined as:
$$
\begin{equation}

\|x-y\|_{2}=\sqrt{\left(x-y\right)^{T}\left(x-y\right)}

\end{equation}
$$

## Matrices

A matrix is a list of vectors (or a table of numbers) which define a linear transformation of vectors:
$$
\begin{equation}

A\in\mathbb{R}^{n\times m}\quad A=\left[\begin{array}{cccc}

a_{11} & a_{12} & \cdots & a_{1m}\\

a_{21} & a_{22} & \cdots & a_{2m}\\

\vdots & \vdots & \vdots & \vdots\\

a_{n1} & a_{n2} & \cdots & a_{nm}

\end{array}\right]

\end{equation}
$$
and the notation $A\in\mathbb{R}^{n\times m}$ means that the matrix $A$ holds $n\times m$ different real items.

The multiplication of a matrix $A\in\mathbb{R}^{n\times m}$ with a vector $x\in\mathbb{R}^m$ (notice the dimensions) is defined as:
$$
\begin{equation}

\left[Ax\right]_{j}=\sum_{i}a_{ji}x_{i}

\end{equation}
$$
where $\left[Ax\right]_{j}$ is the $j$-th index of the resulting vector from the multiplication. If we define $a_i$ to be the $i$-th row of the matrix, such that:
$$
\begin{equation}

A=\left[\begin{array}{ccc}

- & a_{1} & -\\

- & a_{2} & -\\

& \vdots\\

- & a_{n} & -

\end{array}\right]

\end{equation}
$$
then we can write the product of a vector with a matrix more cleanly as:
$$
\begin{equation}

Ax=\left[\begin{array}{c}

a_{1}^{T}x\\

a_{2}^{T}x\\

\vdots\\

a_{n}^{T}x

\end{array}\right]\in\mathbb{R}^{n}

\end{equation}
$$

The multiplication of a matrix $A\in\mathbb{R}^{n\times m}$ with a matrix $B\in\mathbb{R}^{m\times k}$ has the following elements:
$$
\begin{equation}

C_{ij}=\left[AB\right]_{ij}=\sum_{\ell}a_{i\ell}b_{\ell j}

\end{equation}
$$
where $C\in\mathbb{R}^{n\times k}$.

There are a few common families of matrices that we will use, so it will be useful to give them names:

* Matrices with the same number of rows and columns are called _square_ matrices
* Matrices where we can change the order of the indices $A_{ij}=A_{ji}\Rightarrow A^{T}=A$ are called _symmetric_ matrices
* Matrices with non-zero values only on the diagonal $A_{ii}$ are called _diagonal matrices_ and when the whole diagonal is equal to 1, these matrices are denoted as $I$. We can think of $I$ as the identity transformation - for any vector $x$ we get $Ix=x$ and for any matrix $A$ we get $IA=A$
* A matrix $A$ that has a corresponding matrix $B$ such that $AB=BA=I$ is called an _invertible_ matrix, and $B$ is called the _inverse_ of $A$. If $A$ is invertible, it's inverse is unique - because of this we will write the inverse as $A^{-1}$. Also, note that from the definition that $AA^{-1}=A^{-1}A=I$ that $A$ must be square in order to be invertible at all
* An orthogonal matrix $U$ is a matrix whose transpose is it's inverse, i.e. $UU^{T}=U^{T}U=I$

----
**Example: Blockwise Inversion**

Suppose we want to find the inverse of the matrix:

$$\begin{equation}

M=\left[\begin{array}{cc}

A & 0\\

C & D

\end{array}\right]

\end{equation}$$

where $A\in\mathbb{R}^{n\times n}$, $C\in\mathbb{R}^{m\times n}$ and $D\in\mathbb{R}^{m\times m}$, so that $M\in\mathbb{R}^{n+m\times n+m}$. Here we have defined $M$ by it's blocks and will find the inverse with respect to these blocks. So we have to find a matrix such that:

$$\begin{equation}

\left[\begin{array}{cc}

A & 0\\

C & D

\end{array}\right]\left[\begin{array}{cc}

T_{1} & T_{2}\\

T_{3} & T_{4}

\end{array}\right]=I

\end{equation}$$

The important thing to notice is that we are still multiplying matrices by their rules, that means that we multiply $\left[A\quad0\right]$ by $\left[T_{1}\quad T_{3}\right]$, $\left[A\quad0\right]$ by $\left[T_{2}\quad T_{4}\right]$ and so on:

$$\begin{equation}

\left[\begin{array}{cc}

AT_{1}+0T_{3} & \:\cdot\\

\cdot & \:\cdot

\end{array}\right]=\left[\begin{array}{cc}

A & \:0\\

\cdot & \:\cdot

\end{array}\right]\left[\begin{array}{cc}

T_{1} & \:\cdot\\

T_{3} & \:\cdot

\end{array}\right]

\end{equation}$$

so the result will be:

$$\begin{equation}

\left[\begin{array}{cc}

A & 0\\

C & D

\end{array}\right]\left[\begin{array}{cc}

T_{1} & T_{2}\\

T_{3} & T_{4}

\end{array}\right]=\left[\begin{array}{cc}

AT_{1}+0T_{3} & \;AT_{2}+0T_{4}\\

CT_{1}+DT_{3} & \;CT_{2}+DT_{4}

\end{array}\right]

\end{equation}$$

The top left corner ($AT_{1}+0T_{3}$) must be equal to the identity, which only happens if $T_{1}=A^{-1}$. The top right corner must be zero and this only happens if $T_{2}=0$. Let's write this intermediate result:

$$\begin{equation}

\left[\begin{array}{cc}

A & 0\\

C & D

\end{array}\right]\left[\begin{array}{cc}

A^{-1} & 0\\

T_{3} & T_{4}

\end{array}\right]=\left[\begin{array}{cc}

I & \;0\\

CA^{-1}+DT_{3} & \;DT_{4}

\end{array}\right]

\end{equation}$$

Now, in the same manner as above, we see from the bottom right corner that $T_{4}=D^{-1}$, which means we are left with finding $T_{3}$ such that:
$$
\begin{equation}

CA^{-1}+DT_{3}=0

\end{equation}$$

This will only happen if $T_{3}=-D^{-1}CA^{-1}$. So the inverse of $M$ is given by:

$$
\begin{equation}

M=\left[\begin{array}{cc}

A & 0\\

C & D

\end{array}\right]\Rightarrow M^{-1}=\left[\begin{array}{cc}

A^{-1} & 0\\

-D^{-1}CA^{-1} & \;D^{-1}

\end{array}\right]

\end{equation}
$$

In these derivations, we implicitly assumed that $A$ and $D$ are non-singular. If they are singular, then $M$ will not be invertible.

----

The above example is a special case of the following rule:

$$\begin{equation}

M=\left[\begin{array}{cc}

A & B\\

C & D

\end{array}\right]\Rightarrow M^{-1}=\left[\begin{array}{cc}

A^{-1}+A^{-1}BL^{-1}CA^{-1} & \quad-A^{-1}BL^{-1}\\

-L^{-1}CA^{-1} & \quad L^{-1}

\end{array}\right]

\end{equation}$$

where $L=D-CA^{-1}B$. In this case, the assumption is that $A$ and $L$ are non-singular.

## Eigenvalues and Eigenvectors

Every matrix has _characteristic_ _directions_ (or _characteristic vectors_) - the directions that "matter most'' to the matrix. If $A$ is a square matrix, then we call these characteristic directions the _eigenvectors_ of $A$. A vector $u\neq0$ is an eigenvector of $A$ if:

$$\begin{equation}

Au=\lambda u

\end{equation}$$

where $\lambda$ is a scalar that is called the _eigenvalue_ corresponding to the eigenvector $u$. Notice that if $u$ is an eigenvector of $A$, then so is $\tilde{u}=\delta u$ for any $\delta\in\mathbb{R}$:

$$\begin{equation}

A\tilde{u}=A\delta u=\delta Au=\delta\lambda u=\lambda\tilde{u}

\end{equation}$$

so the eigenvalues $u$ are not unique, while the eigenvalues $\lambda$ are unique.

The directions of the eigenvectors _are_ unique (as long as the rank of $A$ is full, which for the purpose of this course means that $A$ is invertible) and if the matrix is also symmetric, the eigenvectors are always orthogonal to each other. This means that for an $n\times n$ symmetric matrix with full rank, there are exactly $n$ such directions. So as long as $A$ is invertible and symmetric, then the $n$ eigenvectors form a basis and we can rewrite the product of a matrix with a vector as:

$$\begin{equation}

Ax=A\sum_{i}\left\langle u_{i},x\right\rangle \cdot u_{i}=\sum_{i}\lambda_{i}\left\langle u_{i},x\right\rangle u_{i}

\end{equation}$$

(I wrote $\left\langle u_{i},x\right\rangle$  instead of $u_{i}^{T}x$ just to make this form easier to read, but either way of writing is valid).

These eigenvectors also allow us to rewrite the form of $A$ directly using the eigenvectors by noticing the following relationship:

$$\begin{align}
Ax & =\sum_{i}\lambda_{i}u_{i}^{T}xu_{i}\nonumber \\
& =\sum_{i}\lambda_{i}u_{i}u_{i}^{T}x\nonumber \\
\Leftrightarrow A & =\sum_{i}\lambda_{i}u_{i}u_{i}^{T}
\end{align}$$

In vector form, this means any symmetrical matrix $A$ can be decomposed as:

$$\begin{equation}
A=ULU^{T}
\end{equation}$$

where $UU^{T}=U^{T}U=I$ and $L$ is a diagonal matrix made up of the eigenvalues of $A$. Furthermore, the rows in $U$ are the eigenvectors corresponding to the eigenvalues in $L$. This is called the _eigenvalue decomposition_ (EVD) of a symmetrical matrix. Notice that if $A$ is invertible, we can also easily find the decomposition of $A^{-1}$:

$$\begin{equation}
I=AA^{-1}\Rightarrow A^{-1}=UL^{-1}U^{T}
\end{equation}$$

## Singular Value Decomposition

In a similar way to the eigenvectors and values from above, there is a generalization to all matrices $A\in\mathbb{R}^{m\times n}$. The _singular value decomposition_ (SVD) of a matrix $A$ always exists and is defined as:
$$\begin{equation}
A=U\Sigma V^{T}
\end{equation}$$
where $U\in\mathbb{R}^{m\times m}$ is an orthogonal matrix, $\Sigma\in\mathbb{R}^{m\times n}$ is a diagonal matrix (the diagonal $\Sigma_{ii}$ is non-zero, everything else is a zero) and $V\in\mathbb{R}^{n\times n}$ is also an orthogonal matrix. The terms $\sigma_{i}=\Sigma_{ii}$ are called the \emph{singular values }of $A$, are unique to $A$ and are always non-negative. The SVD is directly connected to the EVD in the following manner:
$$\begin{align}
AA^{T} & =U\Sigma V^{T}V\Sigma U^{T}=U\Sigma^{2}U^{T}\\
A^{T}A & =V\Sigma U^{T}U\Sigma V^{T}=V\Sigma^{2}V^{T}
\end{align}$$
and now we can clearly see that $U$ are the eigenvectors of $AA^{T}$and $V$ are the eigenvectors of $A^{T}A$.

## Determinant and Trace

The \emph{determinant} of a square matrix $A$ with eigenvalues $\lambda_{1},\lambda_{2},...,\lambda_{n}$ is defined as:
$$\begin{equation}
\text{det}\left(A\right)\equiv\left|A\right|\stackrel{\Delta}{=}\prod_{i}\lambda_{i}
\end{equation}$$
(note that the **determinant doesn't have to be positive** even though we write $\left|A\right|$!). We can think of the determinant as a measure for how much the space is stretched by the transformation that $A$ implies. If $\left|A\right|=0$, $A$ will be called _singular_ and will not be invertible. The term singular originates from the fact that if one of the eigenvalues of the matrix is equal to zero, then there is a direction from which all points are transformed into the origin by the matrix. In turn, there can be no inverse transformation that will move the points from the origin back to their original positions, which is why a singular matrix is not invertible. Two useful properties of determinants are:

* $\left|A^{-1}\right|=\frac{1}{\left|A\right|}$
* If $A$ and $B$ are square matrices, then $\left|AB\right|=\left|A\right|\left|B\right|$

The _trace_ of a square matrix $A$ is defined as:
$$\begin{equation}
\text{trace}\left[A\right]\stackrel{\Delta}{=}\sum_{i}A_{ii}
\end{equation}$$
if the eigenvalues of $A$ are $\lambda_{1},\lambda_{2},...,\lambda_{n}$ (as before), then:
$$\begin{equation}
\text{trace}\left[A\right]=\sum_{i}\lambda_{i}
\end{equation}$$
In addition, trace has the following properties:

* $\text{trace}\left[\alpha A+B\right]=\alpha\text{trace}\left[A\right]+\text{trace}\left[B\right]$
* $\text{trace}\left[ABC\right]=\text{trace}\left[CAB\right]=\text{trace}\left[BCA\right]$


## Positive Semi-Definite Matrices

A square, symmetrical, matrix $A\in\mathbb{R}^{n\times n}$ is called positive semi-definite (PSD) if:
$$
\forall x\in\mathbb{\mathbb{R}}^{n}\qquad x^{T}Ax\ge0
$$
and _positive definite_ (PD) if:
$$
\forall x\neq0\in\mathbb{\mathbb{R}}^{n}\qquad x^{T}Ax>0
$$

There are a few useful characteristics that PD and PSD matrices have, including:

1. A matrix $A$ is PD if and only if it's eigenvalues $\lambda_{1},...,\lambda_{n}$ are all positive ($\forall i\,\lambda_{i}>0$). This also means that a PD matrix is invertible since$\left|A\right|=\prod_{i}\lambda_{i}>0$
2. A matrix $A$ is PSD if and only if it's eigenvalues $\lambda_{1},...,\lambda_{n}$ are all non-negative ($\forall i\,\lambda_{i}\ge0$)
3. A matrix $A$ is PD if and only if it can be decomposed as $A=R^{T}R$ such that $R$ is triangular and invertible. This decomposition is unique, in the sense that there exists only one triangular and invertible matrix $R$ such that $A=R^{T}R$. This decomposition is called the _Cholesky decomposition_
4. A matrix $A$ is PSD if and only if it can be decomposed as $A=R^{T}R$

Also, notice that any PD matrix is also PSD, but the opposite isn't true.

----
**Example: Product of a Matrix and it's Transpose**

Suppose we have a matrix $A\in\mathbb{R}^{n\times m}$. We will show that $A^{T}A$ is PSD for _any_ (real) matrix $A$. We need to show that for any vector $x$:
$$
\begin{equation}
x^{T}A^{T}Ax\ge0
\end{equation}
$$
We begin by noticing that $A^{T}A$ is symmetrical since:
$$
\begin{align}
\left(A^{T}A\right)^{T}=A^{T}A
\end{align}
$$
from the definition of transpose (we transpose and change the order). Now, notice we can write the above as an inner product between two vectors:
$$
\begin{align}
x^{T}A^{T}Ax & =\left(Ax\right)^{T}Ax\nonumber \\
 & =\left\langle Ax,Ax\right\rangle \nonumber \\
 & =\|Ax\|^{2}
\end{align}
$$
A norm of a vector is always non-negative, so we see that $x^{T}A^{T}Ax\ge0$, which means that $A^{T}A$ is a PSD matrix, which is exactly what we wanted to show.

From now on it will be a good idea to remember that for any matrix $A$, both $A^{T}A$ and $AA^{T}$ (you can define $B=A^{T}$ and then you are looking at the matrix $B^{T}B$) are PSD matrices.

----


## Derivatives

Many algorithms include a cost/loss function which we will try to optimize as much as we can. Many times the optimization will be equivalent to finding the minima of the cost function. The simplest (analytical) method to do so when the function is convex/concave, or has a single minima/maxima, is by differentiating the function and equating to 0.

The chain rule for 1D functions is:
$$
\begin{equation}

\frac{\partial f\left(g\left(x\right)\right)}{\partial x}=\frac{\partial f\left(g\left(x\right)\right)}{\partial g\left(x\right)}\frac{\partial g\left(x\right)}{\partial x}

\end{equation}
$$
which you are (hopefully) already comfortable with. However, during this course we will use a lot functions of the form $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$, so we will need to first remind ourselves how to treat the derivatives of these functions.

### Jacobian
The _Jacobian_ of a differentiable function $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ is a matrix with dimensions $m\times n$ and we define it as:
$$
\begin{equation}
\left[J_{x}\left[f\left(x\right)\right]\right]_{ij}\stackrel{\Delta}{=}\frac{\partial\left[f\left(x\right)\right]_{i}}{\partial x_{j}}
\end{equation}
$$
In this sense, the Jacobian is a sort of generalization of the derivative in higher dimensions. If the function is of the form $g:\mathbb{R}^{n}\rightarrow\mathbb{R}$, the transpose of the Jacobian will be a vector that is called the _gradient_:
$$
J_{x}\left[g\left(x\right)\right]^{T}\equiv\nabla g\left(x\right)\stackrel{\Delta}{=}\left[\frac{\partial g\left(x\right)}{\partial x_{1}},\;\frac{\partial g\left(x\right)}{\partial x_{2}},\:...,\:\frac{\partial g\left(x\right)}{\partial x_{n}}\right]^{T}
$$

We will often use a different notation for high-order derivatives, that is closer to the 1D definition of derivatives. In our notation, we will use the transpose of the Jacobian:
$$
\begin{equation}
\frac{\partial f\left(x\right)}{\partial x}\stackrel{\Delta}{=}J_{x}\left[f\left(x\right)\right]^{T}
\end{equation}
$$
In other words, if $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$, then $\frac{\partial f\left(x\right)}{\partial x}$ is an $n\times m$ matrix. This definition aligns with the definition of the gradient such that:
$$
\begin{equation}
\frac{\partial g\left(x\right)}{\partial x}\equiv\nabla g\left(x\right)
\end{equation}
$$
and will make differentiating a bit easier to understand later on.

----
**Example: Gradient of the Norm**

Let's look at the function $g\left(x\right)=\|x\|^{2}=\sum_{i}x_{i}^{2}$. The elements of the gradient of this function will be:
$$
\left[\nabla g\left(x\right)\right]_{i}=\frac{\partial\sum_{i}x_{i}^{2}}{\partial x_{i}}=\frac{\partial x_{i}^{2}}{\partial x_{i}}=2x_{i}
$$
so the whole gradient will be:
$$
\frac{\partial g\left(x\right)}{\partial x}=2x
$$
(I'm switching notations constantly on purpose - this is to get you accustomed with the fact that both ways to write the gradient mean the same thing, one of them just reminds us that the gradient is a vector and not a number).

----

### Chain Rule

Many times we want to find the gradient of $g\left(f\left(x\right)\right)$ where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ and $g:\mathbb{R}^{m}\rightarrow\mathbb{R}$ (in this case we are deriving a _scalar_ $g\left(f\left(x\right)\right)$ by the _vector_ $x$). In this case, the chain rule is:
$$
\begin{equation}
\label{eq:multivar-chain-rule}
\frac{\partial g\left(f\left(x\right)\right)}{\partial x}=\underbrace{J_{x}\left[f\left(x\right)\right]^{T}}_{n\times m}\underbrace{\nabla_{f\left(x\right)}g\left(f\left(x\right)\right)}_{m\times 1}

\end{equation}
$$
As you can see, the derivative $\frac{\partial g\left(f\left(x\right)\right)}{\partial x}$ will be a vector in $\mathbb{R}^{n}$, which makes sense since the vector we are differentiating by, $x$, is in $\mathbb{R}^{n}$.

This notation makes the chain rule look more intimidating than it is, by using the notation of normal derivatives we get:
$$
\begin{equation}
\frac{\partial g\left(f\left(x\right)\right)}{\partial x}=\frac{\partial f\left(x\right)}{\partial x}\frac{\partial g\left(f\left(x\right)\right)}{\partial f\left(x\right)}
\end{equation}
$$
which looks exactly like the normal chain rule. However, the distinction that is easy to see in the more "formal" notation in \eqref{eq:multivar-chain-rule}
is that $\nabla_{f\left(x\right)}g\left(f\left(x\right)\right)$ is a _vector_ and $J_{x}\left[f\left(x\right)\right]$ is a _matrix_; this is important to remember, as in this case the order of multiplication _is_ important, unlike when the function is 1 dimensional.

----
**Example: Gradient of the Norm of a Product**

Let's build on the previous example by looking at the function $g\left(x\right)=\|Ax\|^{2}$, where $A\in\mathbb{R}^{m\times n}$ and $x\in\mathbb{R}^{n}$. In this case, $g\left(y\right)=\|y\|^{2}$ and $f\left(x\right)=Ax$. Using the chain rule, we have:
$$
\frac{\partial g\left(f\left(x\right)\right)}{\partial x}=J_{x}\left[f\left(x\right)\right]^{T}\nabla g\left(y\right)
$$
We already know that $\nabla g\left(y\right)=2y$, so all that remains is to find $J_{x}\left[f\left(x\right)\right]$:
$$
\frac{\partial\left(Ax\right)_{j}}{\partial x_{i}}=\frac{\partial\sum_{k}A_{jk}x_{k}}{\partial x_{i}}=A_{ji}
$$
so we see that $\left[J_{x}\left[Ax\right]\right]_{ij}=A_{ij}$, i.e.:
$$
\frac{\partial Ax}{\partial x}=A^{T}
$$
Using the chain rule, we get:
$$
\frac{\partial\|Ax\|^{2}}{\partial x}=2A^{T}Ax
$$

----


# Probability

Bayesian machine learning is probabilistic in nature. To really understand everything that happens, we must have a good understanding of probability. The following section will be a refresher for some of the key concepts in probability<d-footnote>See Bishop 1.2 and Murphy 2.2, although they also assume that most of the content is known ahead of time.</d-footnote> which we will use throughout the course.

## Discrete Probabilities

In general, we define a discrete probability to be a function $P:\Omega\rightarrow\left[0,1\right]$ such that $\sum_{\omega\in\Omega}P\left(\omega\right)=1$. A random variable is a variable that can take any value from $\Omega$. From now on, we will denote the probability that a random variable $X$ takes on the value $x$ as:
$$
\begin{equation}
P\left(x\right)\stackrel{\Delta}{=}P\left(X=x\right)
\end{equation}
$$
This will greatly shorten the amount we need to write in the future.

Usually there won't be only one variable of interest, so we need to find a way to introduce a probability over the interaction of several random variables. The probability that two random variables $X$ and $Y$ will take on specific values is called the _joint probability_ and is notated by:
$$
\begin{equation}
P\left(x,y\right)\stackrel{\Delta}{=}P\left(X=x,Y=y\right)
\end{equation}
$$
Because of the structure of the probability function, we can always move from the joint probability to one of the _marginal probabilities_ by summing out the other variable:
$$
\begin{equation}
P\left(x\right)=\sum_{y}P\left(x,y\right)
\end{equation}
$$
Of course, if $Y$ takes a specific value, then this may effect $X$ in some manner. We notate this _conditional probability_ as $P\left(x|y\right)$; this new function is also a probability function, i.e.
$$
\begin{equation}
\sum_{x}P\left(x|y\right)=1
\end{equation}
$$
We can think of this behavior on the side of $y$ as a "re-weighting'' of specific values that $X$ may take. The joint probability can be written in terms of this re-weighting as:
$$
\begin{equation}
\label{eq:factorized-joint}
P\left(x,y\right)=P\left(x|y\right)P\left(y\right)
\end{equation}
$$
If the value of $X$ doesn't depend on the value of $Y$ at all (and vice-versa) we will say that the two variables are _independent_. In this case the joint probability takes the form of:
$$
\begin{equation}
P\left(x,y\right)=P\left(x\right)P\left(y\right)
\end{equation}
$$
which is a direct result from the rule given by \eqref{eq:factorized-joint}. We define the conditional probability according to _Bayes' law_:
$$
\begin{equation}
P\left(y|x\right)=\frac{P\left(x,y\right)}{P\left(x\right)}=\frac{P\left(x|y\right)P\left(y\right)}{P\left(x\right)}
\end{equation}
$$

Finally, if we have a random variable $X$ and a random variable $Y$ that partitions the space, then we can rewrite the probability for $X$ under the conditionals of $Y$ - this is also called the _law of total probability_:
$$
\begin{equation}
P\left(x\right)=\sum_{y}P\left(x|y\right)P\left(y\right)
\end{equation}
$$

## Continuous Probabilities

While it is of course important to understand the rules of probability in the discrete case, most of the course we will be dealing with _continuous random variables_; these random variables can take any value in $\mathbb{R}$ or a section of it. If we simply try to scale the definition we saw before to the continuous case, then for any non-zero probability over any bounded section $\Gamma$ of $\mathbb{R}$, we have:
$$
\sum_{x\in\Gamma}P\left(x\right)\ge\sum_{x\in\Gamma}\min_{y\in\Gamma}P\left(y\right)\rightarrow\infty
$$
since there are an infinite number of points in the section $\Gamma$. Clearly, we can't use the same reasoning to describe probabilities over continuous variables. Instead, for any section $\left[a,b\right]\subseteq\mathbb{R}$, we will define:
$$
\begin{equation}
P\left(a\le X\le b\right)\stackrel{\Delta}{=}\intop_{a}^{b}p\left(x\right)dx
\end{equation}
$$
where $p\left(x\right)$ is called the _probability density function_ (PDF) of the variable $X$. Under this logic, the only restrictions on $p\left(\cdot\right)$ are that for any $x\in\mathbb{R}$ $p\left(x\right)\ge0$
and:
$$
\intop_{-\infty}^{\infty}p\left(x\right)dx=1
$$
After defining the PDF, all of the rules we have defined earlier apply, only using integrals instead of sums.

## Expectation

One of the most useful statistics involving probabilities we will need in this course is that of finding the weighted average of functions. The average of some function $f\left(x\right)$ under a probability function $p\left(x\right)$ is called the _expectation_ of $f\left(x\right)$ and is denoted as $\mathbb{E}\left[f\right]$. For a discrete distribution it is given by:
$$
\begin{equation}
\mathbb{E}\left[f\left(x\right)\right]\stackrel{\Delta}{=}\sum_{x}p\left(x\right)f\left(x\right)
\end{equation}
$$
This has a very clear interpretation, since $p\left(x\right)$ sums up to 1: it is the averaging of $f$, weighted by the relative probabilities of the variable $x$. For continuous variables, we exchange the sum
with an integral to get:
$$
\mathbb{E}\left[f\left(x\right)\right]\stackrel{\Delta}{=}\intop_{-\infty}^{\infty}p\left(x\right)f\left(x\right)dx
$$
By definition, the expectation is a _linear operator_, i.e.:
$$
\begin{equation}
\mathbb{E}\left[ax+y\right]=a\mathbb{E}\left[x\right]+\mathbb{E}\left[y\right]
\end{equation}
$$

In either case, if we are given a finite number of points, $N$, sampled independently and identically from the distribution, then the expectation can be approximated as:
$$
\mathbb{E}\left[f\left(x\right)\right]\approx\frac{1}{N}\sum_{i}f\left(x_{i}\right)
$$
At the limit $N\rightarrow\infty$, this approximation is exact.

The mean of the distribution $p\left(x\right)$ is simply the expected value of $x$ itself, i.e.:
$$
\begin{equation}
\mathbb{E}[x]=\intop_{-\infty}^{\infty}xp\left(x\right)dx
\end{equation}
$$
We can of course give the same treatment to joint probabilities:
$$
\mathbb{E}_{x,y}\left[f\left(x,y\right)\right]=\intop_{-\infty}^{\infty}\intop_{-\infty}^{\infty}f\left(x,y\right)p\left(x,y\right)dxdy
$$
Moreover, we can look at the averages according to only one of the marginals of the distribution, i.e.:
$$
\begin{equation}
\mathbb{E}_{x}\left[f\left(x,y\right)\right]=\intop_{-\infty}^{\infty}f\left(x,y\right)p\left(x\right)dx
\end{equation}
$$

here we have added the subscript $x$ to denote that we are averaging over $x$ and not $y$. In this case, the expectation will be a function of $y$, as it is still a free variable. We can also consider _conditional expectations_, that is the weighted average of function over the conditional
expectations:
$$
\begin{equation}
\mathbb{E}\left[f\left(x\right)|y\right]=\intop_{-\infty}^{\infty}f\left(x\right)p\left(x|y\right)dx
\end{equation}
$$

## Variance and Covariance

Many times we would also like to measure how much variability there is to the values of the function $f\left(\cdot\right)$. The _variance_ of $f\left(\cdot\right)$, defined as:
$$
\begin{equation}
\text{var}\left[f\left(x\right)\right]\stackrel{\Delta}{=}\mathbb{E}\left[\left(f\left(x\right)-\mathbb{E}\left[f\left(x\right)\right]\right)^{2}\right]=\mathbb{E}\left[f\left(x\right)^{2}\right]-\mathbb{E}\left[f\left(x\right)\right]^{2}
\end{equation}
$$
measures exactly that. Of course, we can also consider the variance
of the variable itself:
$$
\text{var}\left[x\right]=\mathbb{E}\left[x^{2}\right]-\mathbb{E}\left[x\right]^{2}
$$
Another measure that we will see during the course is the _standard deviation_. The standard deviation of a random variable is defined as:
$$
\sigma_{x}\stackrel{\Delta}{=}\sqrt{\text{var}[x]}
$$

When we have many dependent variables, we may also want to see how much each random variable is effected by the other variables. The _covariance_ measures this and is defined by:
$$
\begin{align}
\text{cov}\left[x,y\right] & \stackrel{\Delta}{=}\mathbb{E}\left[\left(x-\mathbb{E}\left[x\right]\right)\left(y-\mathbb{E}\left[y\right]\right)\right]=\mathbb{E}\left[xy\right]-\mathbb{E}\left[x\right]\mathbb{E}\left[y\right]
\end{align}
$$
Directly from the definition, we can see that the covariance of a
random variable with itself is simply its variance:
$$
\text{cov}[x,x]=\mathbb{E}[x^{2}]-\mathbb{E}[ x^{2}]=\text{var}[x]
$$

## Random Vectors

In many applications we will have many random variables that somehow depend on each other - $x_{1},...,x_{n}$. Usually, it will by much easier to group them together into a vector $\boldsymbol{x}=\left(x_{1},...,x_{n}\right)^{T}$
than to consider them individually. In this case, we will also write out the PDF as a function of a vector, so that:
$$
p:\mathbb{R}^{n}\rightarrow\mathbb{R}_{+}
$$
and we will simply write $p\left(\boldsymbol{x}\right)$ instead of $p\left(x_{1},x_{2},...,x_{n}\right)$. Of course, all of the attributes we have introduced above are also available to random vectors. The only real difference from before is the notation. The expectation of a random vector is defined to also be a vector, where each coordinate is the expectation of the random variable from the same coordinate:
$$
\begin{equation}
\mathbb{E}\left[\boldsymbol{x}\right]\in\mathbb{R}^{n}\ \ \ \ \ \ \ \ \ \mathbb{E}\left[\boldsymbol{x}\right]_{i}=\mathbb{E}\left[x_{i}\right]
\end{equation}
$$

This definition of random vectors allows us to define the _covariance matrix_. For two random vectors $\boldsymbol{x}=\left(x_{1},...,x_{n}\right)^{T}$ and $\boldsymbol{y}=\left(y_{1},...,y_{m}\right)^{T}$, we define the covariance matrix as:
$$
\begin{equation}
\text{cov}[\boldsymbol{x},\boldsymbol{y}]=\mathbb{E}[\left(\boldsymbol{x-}
\mathbb{E}[\boldsymbol{x}]\right)\left(\boldsymbol{y}-\mathbb{E}[\boldsymbol{y}]\right)^{T}]=
\mathbb{E}[\boldsymbol{xy}^{T}]-\mathbb{E}[\boldsymbol{x}]\mathbb{E}[\boldsymbol{y}^{T}]
\end{equation}
$$
the result is a matrix of dimension $n\times m$ (yes, $\boldsymbol{x}$
and $\boldsymbol{y}$ don't have to have the same dimension), with
the elements:
$$
\text{cov}[\boldsymbol{x},\boldsymbol{y}]_{ij}=\text{cov}[x_{i},y_{j}]
$$
as expected. For notational convenience, we may use the definition:
$$
\text{cov}[\boldsymbol{x}]\stackrel{\Delta}{=}\text{cov}[\boldsymbol{x},\boldsymbol{x}]
$$
which is the matrix of the covariances between different variables
in the random vector $\boldsymbol{x}$.

The covariance matrix of a single variable $\boldsymbol{x}$ is a PSD matrix, as we can see by writing the covariance explicitly:
$$
\begin{equation}
\text{cov}[\boldsymbol{x}]=\mathbb{E}[\boldsymbol{xx}^{T}]-\mathbb{E}\left[\boldsymbol{x}\right]\mathbb{E}\left[\boldsymbol{x}^{T}\right]=\mathbb{E}[\left(\boldsymbol{x-}\mathbb{E}[\boldsymbol{x}]\right)\left(\boldsymbol{x-}\mathbb{E}[\boldsymbol{x}]\right)^{T}]
\end{equation}
$$
That is, the covariance is the result of a matrix times it's transpose,
which is a family of matrices we have shown to be PSD.

## Change of Variable

As defined so far, every random variable $X$ comes with it's own PDF, $p_{x}\left(\cdot\right)$. When we have two different random variables $X$ and $Y$, we will have two _different_ pdfs $p_{x}\left(\cdot\right)$ and $p_{y}\left(\cdot\right)$. Sometimes it will be helpful to move from one PDF to the other, if the density of one variable depends on the other. However, it isn't clear how we can do this without violating the elementary conditions that PDFs must satisfy. We can bypass this by working with the _cumulative distribution function_ (CDF) of the random variable, instead of the PDF, defined as:
$$
\begin{equation}
P_{y}\left(y\right)\stackrel{\Delta}{=}P\left(Y\le y\right)=\intop_{-\infty}^{y}p_{y}\left(\tilde{y}\right)d\tilde{y}
\end{equation}
$$
As is maybe obvious, we can get back to the PDF by deriving the CDF. If we have a function $f:X\rightarrow Y$ that maps between the random variables, then<d-footnote>See the [Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function\#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function)
page for the change of variables for a slightly more organized explanation
of what's happening here</d-footnote>:
$$
\begin{equation}
P\left(Y\le y\right)=P\left(f\left(X\right)\le y\right)=P\left(X\in\left\{ x\,|\,f\left(x\right)\le y\right\} \right)
\end{equation}
$$
If $f\left(\cdot\right)$ is invertible, we can further simplify this:
$$
\begin{equation}
P\left(f\left(x\right)\le y\right)=P\left(X\le f^{-1}\left(y\right)\right)=P_{x}\left(f^{-1}\left(y\right)\right)
\end{equation}
$$
Differentiating, we move back to the PDF:
$$
\begin{equation}
p_{y}\left(y\right)\stackrel{\Delta}{=}\frac{\partial}{\partial y}P_{y}\left(y\right)=\frac{\partial}{\partial y}P_{x}\left(f^{-1}\left(y\right)\right)=\frac{\partial f^{-1}\left(y\right)}{\partial y}\frac{\partial}{\partial f^{-1}\left(y\right)}P_{x}\left(f^{-1}\left(y\right)\right)=\frac{\partial f^{-1}\left(y\right)}{\partial y}p_{x}\left(f^{-1}\left(y\right)\right)
\end{equation}
$$
In general, since the PDF is non-negative, we will take the absolute value of the derivative to be sure of the result:
$$
\begin{equation}
p_{y}\left(y\right)=\left|\frac{\partial f^{-1}\left(y\right)}{\partial y}\right|p_{x}\left(f^{-1}\left(y\right)\right)
\end{equation}
$$
The derivative term is a re-normalization of the PDF from the $Y$ space to the $X$ space, where $\partial x=\partial f^{-1}\left(y\right)$ and $\partial y$ are measures of the volume of the $X$ and $Y$ spaces respectively - the term in the derivative can then generally be thought of as re-normalizing the PDF so that it is measured in units of volume of the $X$ space instead of units of volume in the $Y$ space.

The same story unfolds in the multivariate case, with the only catch that now we have to use the Jacobian of the transformation and not a simple derivative. Using the same analogy as before (I know this isn't a proof, but that will be harder and not particularly useful), while the Jacobian measures the \emph{change} of the function in each of it's directions, the determinant of the Jacobian measures the _change in volume_ before and after the function (the same as above). With this knowledge, given a function $f:\boldsymbol{x}\rightarrow\boldsymbol{y}$, the change of variable will be:
$$
p_{y}\left(\boldsymbol{y}\right)=p_{x}\left(f^{-1}\left(\boldsymbol{y}\right)\right)\left|J_{y}\left[f^{-1}\left(\boldsymbol{y}\right)\right]\right|
$$

----
**Example: Change of Variable**

Let's build an example for the change of variable rule. Let $x$ be a uniform random variable in the range $\left[0,\;\alpha\right]$; in other words:
$$
\begin{align}
p_{x}\left(x\right) & =\begin{cases}
\frac{1}{\alpha} & x\in\left[0,\;\alpha\right]\\
0 & \text{otherwise}
\end{cases}
\end{align}
$$
Also, we will define $z=x^{2}$ and want to find $p_{z}\left(\cdot\right)$ in terms of $p_{x}\left(\cdot\right)$.

Notice that in the range $0\le x\le\alpha$ - the range where $p_{x}\left(\cdot\right)$ is non-zero (also called the _support_ of $p_{x}\left(\cdot\right)$) -  the function:
$$
\begin{equation}
z=f\left(x\right)=x^{2}
\end{equation}
$$
is invertible, which means that we can use the change of variable rule. The inverse function in the same range is given by:
$$
\begin{equation}
x=f^{-1}\left(z\right)=\sqrt{z}
\end{equation}
$$
The derivative of the inverse function is the following:
$$
\begin{equation}
\frac{\partial f^{-1}\left(z\right)}{\partial z}=\frac{1}{2}\cdot\frac{1}{\sqrt{z}}
\end{equation}
$$
so the PDF of $z$ can be rewritten in terms of the PDF of $x$ as follows:
$$
\begin{equation}
p_{z}\left(z\right)=\frac{1}{2}\cdot\frac{1}{\sqrt{z}}p_{x}\left(\sqrt{z}\right)=\begin{cases}
\frac{1}{2\alpha}\cdot\frac{1}{\sqrt{z}} & 0\le\sqrt{z}\le\alpha\\
0 & \text{otherwise}
\end{cases}
\end{equation}
$$

----
