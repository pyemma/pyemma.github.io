---
  layout: single
  title: What I Read This Week 2
---
### [Evolving search recommendations on Pinterest](https://medium.com/@Pinterest_Engineering/evolving-search-recommendations-on-pinterest-136e26e0468a)
A post introducing the search work done in Pinterest.
* Initially they use a *Term-Query graph* to generate candidates. In this graph, each term (a single word) is represent a node, as well as the query. Each term node is connected to the query, weighted by the reciprocal of the number of queries that term shows up in. Each query node is also connected to query node, weighted by the relativeness. Most visited queries are recommended.
* They later changed to *Pixie*, a graph based recommendation platform. The graph is build using query and pins. Compared with pervious solution, this solution will not break the query and thus keep the semantic information. **To give better recommendation, semantic information is important.**
* They have further work to utilize embeddings for queries. Is embedding based candidate generation works better than random walk based candidate generation?

### [An Overview of Multi-Task Learning in Deep Neural Networks](http://ruder.io/multi-task/index.html#introduction)
A pretty good blog introducing what MTL is, who is works and some recent works. Here are some points I think most beneficial:
* Using one sentence to explain MTL: "By sharing representation among related tasks(leveraging different domain-specific knowledge), the generalization of model is improved."
* Why MTL work:
  - Our training data will always contains some noise. Training a single goal on the data will easily get overfit. However, training on multiple tasks simultaneously will help cancel out this noise (**Data Augmentation**).
  - By training on multiple tasks and sharing the same representation at the same time, the model will try to find a more general hypothesis that would work for all tasks (**Regularization**).
  - Some feature combination might be pretty complex in one task and hard to let the model to capture, but easy in the other one. MTL will help sharing this info between tasks(**Feature Engineering**).
* There are mainly two form of MTL:
  - **Hard Parameter Sharing**: Different tasks will have several same layers at the lower level, and have their own layers at higher level. A common use-case is that, when can use the bottom layers in VGG, and then train our own layer on our task.
  - **Soft Parameter Sharing**: Different tasks will have their own model, but each model can constraint each other to not differ too much.
  - Currently, **Hard Parameter Sharing** is still very popular, but **Soft Parameter Sharing** is more promising as it let the model to learn what to share.

### [Welcoming the Era of Deep Neuroevolution](https://eng.uber.com/deep-neuroevolution/)
Uber AI Lab's work on using Genetic Algorithm instead of SGD to optimize DNN on reinforcement learning tasks.
* GA can produce comparatively similar result as SGD.
* They purposed a method to smartly guide the mutation to put more attention on the sensitive feature, to solve the problem GA has when dealing with large networks.
* They also purposed a method to enforce the exploration, which they try to have a population of candidates that act differently from each other as much as possible (unlikely to trapped in local minima).

### [Effective Modern C++]
Mainly read the smart pointer part.
* `unique_ptr` performs similar to the old fashion raw pointer. It indicates a exclusive ownership to the object it manages, thus it can only be moved not be copied. We can specify a custom deleter to a unique pointer, and the deleter would become part of the unique pointer's type. It is very convenient to covert a unique pointer to a shared pointer.
* `shared_ptr` performs similar to the garbage collection in Java. It indicates a shared ownership to the object. The underlay mechanism is that each shared pointer will create a control block that would keep the reference count and other data. Since there is a separate object holding all extra info, the deleter we passed to shared pointer will not become a part of its type. Remember not using the raw pointer to initialize a shared pointer, as it is pretty dangerous and we might result in free the object multiple times. This is extremely the case when we are working with the `this` pointer. To be able to safely create a shared pointer from `this`, use `enable_shared_from_this` template.
* `enable_shared_from_this` uses *Curiously Recurring Template Pattern (CRTP)* (to be add more details later).
* `weak_ptr` is like `shared_ptr`, but it does not effect the reference count on the object, and the object it is pointing to might be destroyed. The use case for `weak_ptr` can be **cacheing**, **observer lists** and the prevention of `shared_ptr` cycles.  
