---
layout: post
title: Overview
description: Overview of the Bayesian machine learning course and material introduced in each part.

---

There is a lot of interest in machine learning all over the world, in all of its forms and while there is a lot of work out there to do with  Bayesian machine learning,  it is sometimes considered a daunting subject (probably because of the math). This leads to many introductory courses in ML to take completely different directions, without even mentioning what "Bayesian ML" even is.


The purpose of the Bayesian ML (BML) course in the Hebrew University is to give a good basis for probabilistic ML, and specifically to build intuition around Bayesian approaches. The course is targeted towards students early in their studies of ML, but I think there're a lot of hidden gems that even graduate students can extract useful knowledge. I know of a few such cases, including myself. 

The pages in this blog are transcripts from the lessons taught in the BML course. Hopefully, they will help someone out there.

## The Narrative 

### The Basics

We spent a good amount of time deciding the "story" that should be told. Obviously, the first step is to explain what the Bayesian philosophy is and why it is different from other approaches. Following this, we introduce the Gaussian distribution - the most prominent distribution in statistics - and how to estimate the parameters of the Gaussian distribution, classically and by applying the Bayesian approach.

### The Linear Regression Phase

Using the Gaussian distribution as a foundation is crucial, since many classical algorithms are based on it. This is particularly true for linear regression, the first "real" part of the course. Linear regression is both easy to understand _and_ surprisingly flexible. Additionally, it allows for simple demonstrations of how _priors_, the building-blocks of the Bayesian philosophy, can be used to great benefit and what happens if they are badly chosen. Linear regression is also one of the rare examples where the _evidence_ of a model can be calculated, which is a neat method for model selection.

Simple priors paired with linear regression are a direct link to kernel methods, which are much more powerful than simple regression. This link furthers the discussion of priors from priors on the parameters of the linear regression, to priors _over functions_ - an elegant way to start talking about Gaussian processes.

### Classification

Regression isn't the only task that interests people - in fact, classification might be even more interesting for most people. Using probabilistic approaches there are two options for classification: generative or discriminative. Both of these options are interesting, but generative classification is a particularly powerful case of Bayes' law.

### Moving Away from Gaussians

While the Gaussian distribution is a useful tool, it is not very expressive. In this part, we will try to move away from the Gaussian distribution to use _mixtures of Gaussians_, which are much more flexible. Moving from Gaussians to mixtures of them raises the question of how the prior is chosen and which solutions to choose.
