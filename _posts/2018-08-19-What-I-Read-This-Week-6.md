---
  layout: single
  title: What I Read This Week 6
  categories:
    - reading notes
  tags:
    - model interpreting
---

#### Model Interpreting
* [Ideas on interpreting machine learning](https://www.oreilly.com/ideas/ideas-on-interpreting-machine-learning)
* Residual analysis can help understand which part of the data model is doing wrongly and thus find solution to improve it. However it's straight forward to apply this method if the problem is a pure regression problem, which is actually not a common practice in industry. How to do this with for logistic regression? (devariance?)
* A bunch of other method: surrogate model (train a simpler model on the output of the complexity model, which is similar to distill info of a neural network into a BDT), [LIME](https://arxiv.org/pdf/1602.04938v1.pdf), maximum activation analysis (pair with LIME is better), variance importance (different way to compute the importance, pervade in tree model), sensitivity analysis, leave one covariance out, tree interpreter.
