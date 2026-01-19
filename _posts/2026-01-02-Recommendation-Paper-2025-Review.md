---
title: My 2025 Recommendation System Paper Summary
date: 2026-01-02
author: pyemma
categories: [Machine Learning]
tags: [recommendation-system, llm4rec]
toc: true
math: true
---

In this post, I would like to share some insights from the paper I have read in year 2025 and summarize some trends over the year.

### The *One*

The best work I enjoyed this year is the *One*-series from Kuaishou, such as *OneRec*, *OneSearch*, *OneRec v2*, etc. From high level, the *One*-series use a **single** generative recommendation model (which is usually use encoder-decoder or some variation as the backbone) to replace the widely used **cascading multi-stage architecture** in industry recommendation system. The primary reason that I enjoy this work is that:

1. It established a new paradigm for recommendation model, challenging the classical architecture that has been used in industry for years, with the claim to improve MFU on the latest hardware
2. By leveraging semantic ids, it suggests one solution how recsys could be unified with LLM (pre-training) and harvest the advance in this realm (e.g. different attention mechanism)
3. Integrate reinforcement learning as part of post-training to facilitate the optimization of different business goals

There is definitely long way for this new paradigm to really replace the old one (there is rumor that this work is not launched to full traffic in Kuaishou and only adopted for partial traffic in the retrieval stage due to the cost) and I don't believe *One*-series is the ultimate status of how the LLM4Rec would be like. However, I believe in this direction and several technique used in this work would become the cornerstone for the future.

### User-Item interactions, to be or not to be

In one tech-salon hosted by Kuaishou this year, one audience asked a specific question regarding the *OneRec* work: **how does it learns the user-item feature interactions, which has been proven to be critical in the legacy recsys models**. This question puzzles me as well. If you look at some common generative recommender model implementation, it seems that we only have item-item interactions as the attention is computed over the user-behavior sequence.

My personal intuitive explanation to this is that the user-item interaction is learnt implicitly from the sequence, as the sequence itself encodes users' behavior signal directly. Thus, each item-item interaction is conditioned on users, which bias to kind of item-item interaction model could learn. To me, this is a type of weak, or implicit user-item interactions.

But of course, there are some works to make user-item interaction to be explicit in the transformer style of model. *Rankmixer* and *OneTrans* work from Bytedance is great example from this year. The key idea in these work is to tokenize user features via some algorithm (e.g. feature grouping) and use them as query during the attention to interact with the item sequence.

IMHO, user-item interaction is critical for recsys (and it is a form of weak attention as well if you think how it was computed). I believe it would become a default rule-of-thumb in the future.

### Grand Unified ~~Theory~~ Model

In physics, there is *Grand Unified Theory* that tries to unify gravity, electromagnetic force, strong interaction and weak interaction into a single theory; in recsys, there is also similar effort, that tries to unify different domain/surface/tasks into a single model. Such a model is usually called a *Foundation Model*.

Having a *Foundation Model* has several benefits, such as reduced maintenance complexity, better generalizability due to larger volume of data and better task performance (if done correctly). *Foundation Model* is usually several billion in terms number of parameters which makes them pretty expensive to inference in online. A common strategy is to use them as *information compressor* to pre-compute the embedding which would be used for downstream models; another one is to use the knowledge distillation (offline or online) to transfer knowledge to student models. *Meta Lattice* and *External Large Foundation Model* work from Meta Ads are good examples.

> If you find this post helpful, feel free to scan the QR code below to support me and treat me to a cup of coffee
{: .prompt-tip }

![Thank You](/assets/qr%20code.png){: width="300" height="300" }

### Reference

- [OneRec](https://arxiv.org/abs/2502.18965)
- [OneRec V2](https://arxiv.org/abs/2508.20900)
- [OneSearch](https://arxiv.org/abs/2509.03236)
- [RankMixer](https://arxiv.org/abs/2507.15551)
- [OneTrans](https://arxiv.org/abs/2510.26104)
- [Meta Lattice](https://arxiv.org/abs/2512.09200)
- [External Large Foundation Model](https://arxiv.org/abs/2502.17494)