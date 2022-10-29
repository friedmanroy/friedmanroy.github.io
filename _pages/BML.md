---
layout: page
title: BML
permalink: /BML/
description: Summaries of material from the Bayesian Machine Learning course taught in the Hebrew University
nav: true
nav_order: 2
horizontal: false
pagination:
  enabled: true
  per_page: 20
  sort_field: date
  sort_reverse: false
---

<!-- {% for summary in site.BML %}
  <h2>
    <a href="{{ summary.url }}">
      {{ summary.title }}
    </a>
  </h2>
{% endfor %} -->

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
        {{ read_time }} min read &nbsp; &middot; &nbsp;
      </p>

    </li>

    {% endfor %}
  </ul>
{% include pagination.html %}
</div>
