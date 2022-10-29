---
layout: page
title: Bayesian Machine Learning
permalink: /BML/
nav: true
nav_order: 2
horizontal: false
---


The following posts contain the summaries of lessons in the Bayesian Machine Learning course taught at the Hebrew University. These summaries don't contain all of the material, but are pretty comprehensive all on their own.


---


<div class="post">

  <ul class="post-list">
    {% for post in site.BML %}

    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}

    {% assign year = post.date | date: "%Y" %}

    <li>
      <h3>
        <a class="post-title" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </h3>

      <p>{{ post.description }}</p>
      <p class="post-meta">
        {{ read_time }} min read &nbsp;
      </p>

    </li>

    {% endfor %}
  </ul>
</div>
