---
layout: page
title: BML
permalink: /BML/
description: Summaries of material from the Bayesian Machine Learning course taught in the Hebrew University
nav: true
nav_order: 2
horizontal: false
---

{% for summary in site.BML %}
  <h2>
    <a href="{{ summary.url }}">
      {{ summary.title }}
    </a>
  </h2>
{% endfor %}
