var store = [{
        "title": "Ant Cheat Sheet",
        "excerpt":"This is a simple document introducing how to write the build.xml of Ant for your projects.Each Ant build.xml has a &lt;project&gt; element as the root element, this element can have a name attribute, which specify the name of this project; a basedir element, which determines the root path during the...","categories": [],
        "tags": ["project","tools"],
        "url": "https://pyemma.github.io/Ant-Cheat-Sheet/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "[Lintcode] Triangle Count",
        "excerpt":"Given an array of integers, how many three numbers can be found in the array, so that we can build an triangle whose three edges length is the three numbers that we find?The most naive solution is to enumerate all triples and then check if they can construct a triangle...","categories": [],
        "tags": ["algorithm","interview"],
        "url": "https://pyemma.github.io/Lintcode-Triangle-Count/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Simple Bloom Filter Implementation Part I",
        "excerpt":"Recently, I’m studying some basic concepts in distributed system. The materials I’m using is Distributed Systems Concepts. I know that simply reading the book is far not enough: the concepts are abstract, but we need to handle partical problem. So I decided to do some simple projects, using some existing...","categories": [],
        "tags": ["project","distributed system","data structure"],
        "url": "https://pyemma.github.io/Simple-Bloom-Filter-Implementation-Part-I/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Simple Bloom Filter Implementation Part 2",
        "excerpt":"IntroductionIn the last blog, we introduced the initial version of bloom filter. In the first implementation, we only tested our bloom filter on built in type String. This time, we tested in against a custom class Person. The idea is simple: we use thrift to define our custom data structure....","categories": [],
        "tags": ["project","distributed system","data structure"],
        "url": "https://pyemma.github.io/Simple-Bloom-Filter-Implementation-Part-2/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "[Hackerrank] Cut the tree",
        "excerpt":"The problem is available here. The idea is to use postorder traversal of the tree to enumerate each possible remove of edge, and return the sum of the sub-tree to its parent node. This problem represent the tree as a acyclic undirected graph, which is a quite wired representation.public class...","categories": [],
        "tags": ["algorithm","interview"],
        "url": "https://pyemma.github.io/Hackerrank-Cut-the-tree/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "[Lintcode] Count of Smaller Number before itself",
        "excerpt":"Give you an integer array (index from 0 to n-1, where n is the size of this array, value from 0 to 10000) . For each element Ai in the array, count the number of element before this element Ai is smaller than it and return count number array.In this...","categories": [],
        "tags": ["algorithm","interview"],
        "url": "https://pyemma.github.io/Lintcode-Count-of-Smaller-Number-before-itself/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "[Lintcode] Interval Sum II",
        "excerpt":"Given an integer array in the construct method, implement two methods query(start, end) and modify(index, value): For query(start, end), return the sum from index start to index end in the given array. For modify(index, value), modify the number in the given index to value public class Solution { private TreeNode...","categories": [],
        "tags": ["algorithm","interview"],
        "url": "https://pyemma.github.io/Lintcode-Interval-Sum-II/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "[CodeFights] NumberGameVI",
        "excerpt":"The problem is available here. A relatively straight forward game problem. If the current person want to win, the only requirement is that there exist a move can make B lose. And if there is no move can make B lose, then A will lose. Translate this into recursion and...","categories": [],
        "tags": ["algorithm","interview"],
        "url": "https://pyemma.github.io/CodeFights-NumberGameVI/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Design Pattern Summarization",
        "excerpt":"Design Pattern SummarizationStrategy PatternDefines a family of algorithms, encapsulate each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.[Image: file:///-/blob/eDDAAAJA4tZ/2DMfZKq387ZdSBlQHdKHXg]Use composition instead of inheritance to separate the detailed actions from the clients code. The detailed actions can be determined or changed in run...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Design-Pattern-Summarize/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Clean Code Summarize",
        "excerpt":"Here is just some tips I think is most useful for guiding me to write better code from book Clean Code: A Handbook of Agile Software Craftsmanship.##NamingWe cannot avoid naming things well we are writing code, a good name can considerably increase the readability of the code. We even does...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Clean-Code-Summarize/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Math Equation",
        "excerpt":"Use $$ to write math equationUse \\begin{equation} to write math equation\\begin{equation}\\sum_{n=1}^\\infty 1/n^2 = \\frac{\\pi^2}{6}\\end{equation}","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Math-Equation/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 1",
        "excerpt":"This is the first post for the series reinforcement learning. The main source for the entire series is here. The post mainly focus on summarizing the content introduced in the video and slides, as well as some of my own understanding. Any feedback is welcomed.In this post, we will talk...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-1/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 2",
        "excerpt":"In the last post, we introduced the definition of Markov Decision Process and Bellman Equation. Now, if you are given the states \\( S \\), action $A$, transition matrix $P$, rewards $R$ and discounting ratio \\( \\gamma \\), how would you come up with a solution for this MDP? i.e....","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-2/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 3",
        "excerpt":"In this lesson, we will learn about what to do when we have no knowledge about the MDP. In the last lesson, we learnt about how to solve a MDP when we have full information about it (e.g. $P$, $R$). When we don’t have enough information, the Bellman Equation won’t...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lession-3/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 4",
        "excerpt":"In this lecture, we learn how to solve an unknown MDP. In the last lecture, we introduced how to calculate the value function given a policy. In this one, we will try to find the optimize policy by ourselves.Mento Calro Policy IterationIn the Lesson 2, we mentioned how to solve...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-4/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 5",
        "excerpt":"In this post, we are going to look into how can we solve the real world problem with a practical way. Think of the state value function $v(s)$ or the action value function $q(s, a)$ we mentioned before. If the problem has a really large state space, then it would...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-5/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 6",
        "excerpt":"In the pervious we use a model to approximate the state value/action value function. In this post, we are going to learn how to directly parameterize a policy, which means we would directly get the probability of each action given a state:In this case, we are not going to have...","categories": [],
        "tags": ["reinforcement learning"],
        "url": "https://pyemma.github.io/Reinforcment-Learning-Lesson-6/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 7",
        "excerpt":"In the pervious notes, we are all using model-free reinforcement learning method to find the solution for the problem. Today we are going to introduce method that directly learns from the experience and tries to understand the underlaying world.From Lesson 1 we know that a MDP can be represent by...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-7/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "Reinforcement Learning Lesson 8",
        "excerpt":"This is the last lesson for the entire reinforcement learning, and in this lesson we will learn something related to exploit and explore. In machine learning service, like recommendation service, there is always a trade off between exploit and explore. Exploit means we are always choosing the best given the...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/Reinforcement-Learning-Lesson-8/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"},{
        "title": "DQN In Practice",
        "excerpt":"Recently I have been working on Deep-Q-Learning and apply it to some interesting AI games. In this post, I would like to give a brief introduction to how I implemented the Deep-Q-Learning, as well as lots of learning along the way.What is DQNTo understand DQN, we need first know is...","categories": [],
        "tags": [],
        "url": "https://pyemma.github.io/DQN-In-Practice/",
        "teaser":"https://pyemma.github.io/assets/violet.jpg"}]
