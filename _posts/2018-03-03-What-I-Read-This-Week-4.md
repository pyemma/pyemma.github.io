---
  layout: single
  title: What I Read This Week 4
---
### [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)
A really good writing on helping understand why neural network can approximate any functions, without too much illustrate on complex mathematic theory.

The main idea the author used is that, for any activation function used in neuron, we can make it has a pretty large weight and bias and it would be like a function that, before some value it would be 0 and after it would be 1. We can have two neurons, manipulate the weight and bias to get a step function. The range of this step function could be pretty small, together with the idea of calculus, we can use millions of such step function to approximate almost all functions. Here is an example from the original blog to show this idea. The parameter \\(s \\) is get by \\( s = -b/w \\), and \\( h \\) is a parameter used to control the hight of the step function. For example, the neuron \\(s = 0.4 \\) and \\( s = 0.6 \\) is a pair, the \\( h \\) for the first one is \\( -1.2 \\) and for the second one is \\( 1.2 \\). Thus, they worked together, we get a step function that starts at \\( 0.4 \\), jump to \\( -1.2 \\), and back at \\( 0.6 \\).

![Neural Network Approximate Any Function](/assets/nn_approx.png)

### [Understanding Capsule Networks — AI’s Alluring New Architecture](https://medium.freecodecamp.org/understanding-capsule-networks-ais-alluring-new-architecture-bdb228173ddc)
Another very good tutorial blog on capsule neural network, the author also provide a visualization tool that we can play with.
