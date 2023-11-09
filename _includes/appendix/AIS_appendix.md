
# Appendix
<d-byline></d-byline>

## **A.1 Importance Sampling**

We know how to calculate $\tilde{p}(x)$, but don't know how to sample from it. The simplest solution for calculating $\overline{f}$ and $Z$ is through what is called _importance sampling_.


Start by choosing a simpler distribution $q(x)$ whose normalization is completely known _and_ is easy to sample from<d-footnote>Also, you have to make sure that the support of $p(x)$ is contained in the support for $q(x)$!</d-footnote>. Then:

$$
\begin{align}
\mathbb{E}_{x\sim p}[f(x)]&=\intop p(x)f(x)dx\\
&=\intop \frac{p(x)}{q(x)}f(x)q(x)dx\\
&=\mathbb{E}_{x\sim q}\left[\frac{p(x)}{q(x)}\cdot f(x)\right]
\end{align}
$$
<br>

Using $q(x)$, we somehow magically moved the difficulty of sampling $x$ from $p(x)$ to the much simpler operation of sampling $x$ from $q(x)$! The expectation can now be approximated using a finite number of samples. Let $w(x)=p(x)/q(x)$ and generate $M$ samples from the distribution $q(x)$ such that:

$$
\begin{equation}
\mathbb{E}_{x\sim p}\left[f(x)\right]\approx \frac{1}{M}\sum_{i:\ x_i\sim q}^M w(x_i)\cdot f(x_i)
\end{equation}
$$
<br>

But there's a problem: we don't really know how to calculate $p(x)$ (since we don't know $Z$), only $\tilde{p}(x)$. Fortunately, we can also estimate $Z$ for the same price! Denote $\tilde{w}(x)=\tilde{p}(x)/q(x)$, then:

$$
\begin{align}
Z&=\intop \tilde{p}(x)dx=\intop\frac{\tilde{p}(x)}{q(x)}q(x)dx\\
&=\intop \tilde{w}(x)q(x)dx\\
&=\mathbb{E}_{x\sim q}\left[\tilde{w}(x)\right]\\
&\approx \frac{1}{M}\sum_{i:\ x_i\sim q}^M\tilde{w}(x_i)
\end{align}
$$
<br>

So, our estimate of $\overline{f}$ is given by:

$$
\begin{equation}
\mathbb{E}_{x\sim p}\left[f(x)\right]\approx \frac{1}{\sum_i\tilde{w}(x_i)}\cdot\sum_{i:\ x_i\sim q}^M \tilde{w}(x_i)\cdot f(x_i)
\end{equation}
$$
<br>

The $w(x)$ (and their unnormalized versions) are called _importance weights_ as for each $x_i$ they capture the relative importance between $p(x_i)$ and $q(x_i)$. 

At the limit $M\rightarrow\infty$, the above approximation becomes accurate. Unfortunately, when $M$ is finite, this estimation is biased and in many cases can be very misspecified.

<br>
<br>
## **A.2 AIS Importance Weights**
<div class="l-gutter" style="margin-bottom:-1000px;margin-top:30px;color:#aaaaaa">
  <b>
  Prerequisites:
  </b>
  <ul style="margin-left:-20px">
  <li style="margin:0px"> Markov chains</li>
  <li style="margin:0px"> Detailed balance</li>
  <li style="margin:0px"> Importance sampling</li>
  </ul>
</div>

To properly understand the construction of the importance weights in AIS, we are going to need to be more precise than my explanation in the main body of text.

So, as usual, we have a target distribution $p(x)=\pi_T(x)/Z_T$ and a proposal distribution $q(x)=\pi_0(x)/Z_0$. In between these two distributions, we have $T-1$ intermediate distributions unnormalized distributions, $\pi_1(x),\cdots,\pi_{T-1}(x)$. The missing piece in the original body of text is the fact that we have $T$ different _transition operators_ that are invariant to the different distributions, which we will call $\mathcal{T}_t(x\rightarrow x')$ for an operation that starts at $x$ and ends at $x'$. In practice, we can think of these as the transition probabilities in a Markov chain.

What do I mean by "invariant transition operators"? Well, these will be our sampling algorithms, so Langevin dynamics on the $t$-th distribution, $\pi_t(x)$. The "invariant" part just means that this transition operator maintains _detailed balance_ with respect to the distribution $\pi_t(x)$:

$$
\begin{equation}
\mathcal{T}_t(x\rightarrow x')\frac{\pi_t(x)}{Z_t}=\mathcal{T}_t(x'\rightarrow x)\frac{\pi_t(x')}{Z_t}
\end{equation}
$$
<br>

As long as $\mathcal{T}_t(x\rightarrow x\')$ has this property for every possible pair of $x$ and $x\'$, it can be used in AIS.

Now, recall that the sampling procedure in AIS was carried out as follows:
<center>
sample $x_0\sim \pi_0$
</center>
<center>
generate $x_1$ using $\mathcal{T}_1(x_0\rightarrow x_1)$
</center>
<center>
$\vdots$
</center>
<center>
generate $x_T$ using $\mathcal{T}_T(x_{T-1}\rightarrow x_T)$
</center>

<br>
This procedure describes a (non-homogeneous) Markov chain, with transition probabilities determined according to $\mathcal{T}_t$.

In the scope of this Markov chain, we can talk about the forward joint probability (starting at $x_0$ and moving to $x_T$) and the reverse joint probability (starting at $x_T$ and going back). At it's root, AIS is just importance sampling with the reverse joint as the target and the forward as the proposal. Mathematically, define:

$$
\begin{align}
\pi(x_0,\cdots,x_T)&=\pi_T(x_T)\cdot\mathcal{T}_T(x_T\rightarrow x_{T-1})\cdots \mathcal{T}_1(x_1\rightarrow x_0)\\
q(x_0,\cdots,x_T)&=q(x_0)\cdot\mathcal{T}_1(x_0\rightarrow x_1)\cdots \mathcal{T}_T(x_{T-1}\rightarrow x_T)
\end{align}
$$
<br>

Of course, we never actually observe $T_t(x_t\rightarrow x_{t-1})$, only the opposite direction. How can we fix this? Well, using detailed balance:

$$
\begin{equation}
\mathcal{T}_t(x_t\rightarrow x_{t-1})=\frac{\pi_t(x_{t-1})}{\pi_t(x_t)}\cdot\mathcal{T}_t(x_{t-1}\rightarrow x_t)
\end{equation}
$$
<br>

This neat property allows us to write the full form of the importance weights<d-footnote>Getting to the last line requires rearranging the terms and using the equation above for the reverse transition, but this post is already pretty long and I don't think adding that math particularly helps here... Everything cancels out nicely and we get the form for the importance weights as in the main text!</d-footnote>:

$$
\begin{align}
w=&\frac{\pi(x_0,\cdots,x_T)}{q(x_0,\cdots,x_T)}\\
&=\frac{\pi_T(x_T)}{q(x_0)}\cdot\frac{\mathcal{T}_T(x_T\rightarrow x_{T-1})\cdots \mathcal{T}_1(x_1\rightarrow x_0)}{\mathcal{T}_1(x_0\rightarrow x_1)\cdots \mathcal{T}_T(x_{T-1}\rightarrow x_T)}\\
&=Z_0\cdot \frac{\pi_1(x_0)}{\pi_0(x_0)}\cdot\frac{\pi_2(x_1)}{\pi_1(x_1)}\cdots\frac{\pi_T(x_{T-1})}{\pi_{T-1}(x_{T-1})}
\end{align}
$$
<br>

These importance weights are exactly the same as those defined in the main body of text, but their motivation is maybe clear now?

The important point is that the proposal distribution creates a path from $x_0$ to $x_T$ while the "true target distribution" is the path from $x_T$ to $x_0$. So the importance weighting is now the forward path $\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)$ as a simpler alternative to the reverse path $\stackrel{\leftarrow}{\mathcal{T}}(x_T\rightarrow x_0)$.

To hammer this point home, the normalization constant for $\pi_T(x)$ can be found by taking the expectation with regards to the forward paths:
$$
\begin{equation}
Z_T=\mathbb{E}_{\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\left[w\right]=\mathbb{E}_{\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\left[\frac{\pi_T(x_T)\stackrel{\leftarrow}{\mathcal{T}}(x_T\rightarrow x_0)}{q(x_0)\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\right]
\end{equation}
$$
<d-byline></d-byline>

That was... probably hard to follow. Hopefully I got some of the message across - there is a Markov chain that goes from $q(x)$ to $\pi_T(x)$ and the reverse of it. If you understood that, and are comfortable with importance sampling, then you're fine. It'll sink in if you think about it a bit more.

This is a neat mathematical trick, though. Theoretically, it is no different than standard importance sampling, we just defined weird proposal and target distributions. Transforming the a simple distribution to something close to the target, though, that's the core of it.

If you read this far, well, I commend you. Good luck using AIS!