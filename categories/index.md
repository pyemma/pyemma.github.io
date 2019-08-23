---
layout: single 
title: Categories 
comments: false
---
 
{% capture site_categories %}{% for category in site.categories %}{{ category | first }}{% unless forloop.last %},{% endunless %}{% endfor %}{% endcapture %}
<!-- site_categories: {{ site_categories }} -->
{% assign category_words = site_categories | split:',' | sort %}
<!-- category_words: {{ category_words }} -->
 
<div id="categories">
  <ul class="category-box inline">
  {% for item in (0..site.categories.size) %}{% unless forloop.last %}
    {% capture this_word %}{{ category_words[item] | strip_newlines }}{% endcapture %}
    <li><a href="#{{ this_word | cgi_escape }}">{{ this_word }} <span>{{ site.categories[this_word].size }}</span></a></li>
  {% endunless %}{% endfor %}
  </ul>
 
  {% for item in (0..site.categories.size) %}{% unless forloop.last %}
    {% capture this_word %}{{ category_words[item] | strip_newlines }}{% endcapture %}
  <h2 id="{{ this_word | cgi_escape }}">{{ this_word }}</h2>
  <ul class="posts">
    {% for post in site.categories[this_word] %}{% if post.title != null %}
    <li itemscope><span class="entry-date"><time datetime="{{ post.date | date_to_xmlschema }}" itemprop="datePublished">{{ post.date | date: "%B %d, %Y" }}</time></span> &raquo; <a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}{% endfor %}
  </ul>
  {% endunless %}{% endfor %}
</div>
