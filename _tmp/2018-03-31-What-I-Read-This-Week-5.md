---
  layout: single
  title: What I Read This Week 5
  tags:
    - reading notes
---
[Google and Uber’s Best Practices for Deep Learning](https://medium.com/intuitionmachine/google-and-ubers-best-practices-for-deep-learning-58488a8899b6)
A good post introducing how **Uber** and **Google** apply deep learning in real world (not in academic). The post illustrates some highlight of the two machine learning platform build by these two giants: [Michelangelo](https://eng.uber.com/michelangelo/) and [TFX](http://www.kdd.org/kdd2017/papers/view/tfx-a-tensorflow-based-production-scale-machine-learning-platform).

For these two platform, there share some common idea: *enforce the share of knowledge across the company*. For example, the all provide feature store, which is shared across the company and every team can reuse the feature easily. Also, the meta-data about model is also stored so that everyone can learn how model is trained, how's the performance and might apply similar model design to their own problem.

The TFX is relatively more complex than Michelangelo (because Google is larger in scale right :) ). Some unique traits included in TFX that impresses me a lot:
* TFX emphasize a lot on data management, it provides a mechanism to monitor the distribution and statistic of the data. This will help engineer understand the data as well as detecting some abnormality in the data.
* TFX also provide transfer-learning by making warm-up from model trained on common features pretty easy.

[TensforFlow Eager Execution](https://www.tensorflow.org/programmers_guide/eager)
Helpful for model debugging.
