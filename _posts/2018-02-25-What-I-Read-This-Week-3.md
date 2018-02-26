---
  layout: single
  title: What I Read This Week 3
---
### [Explicit vs. Implicit Recommenders](https://medium.com/the-graph/insights-from-an-evening-with-recommender-systems-experts-ab44d677dc5e)
* Never use RMSE (or other metrics only take explicit feedback (e.g. rating)) as measurement for your recommendation system
* Implicit feedback (e.g. click/view) is far more valuable than explicit one
* Netflix is not using **star** anymore as an effort to remove explicit feedback

This is a pretty interesting argument and I would like to put a question mark on it. Plan to do some experiment to justify this argument.

### [Introduction to Learning to Trade with Reinforcement Learning](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)
A good post on introducing using reinforcement learning techniques to do trading. The author opens with some background and basic concept in trading, then comes with the limitation of using supervised learning method and current approach. Explained how we can model the trading into reinforcement learning, what kinds of environment, state, actions, rewards, etc. And then stated the beneficial to use reinforcement learning.

* It's pretty hard to make money by predicting the price only through supervised learning. Even though we could accurately predict the price next time, we still need to face the problem of **liquidity available**, **network latency** and **fees**. To make money by supervised learning, we need to accurate predict the large volume of price change or a long time period, or smartly manage our orders and fee, which all is very challenge.
* Another problem with supervised learning is that its not indicate a policy we should execute. You have to come up with a rule based policy.
* For the reinforcement learning based strategy, we have several beneficial points:
  - *Learned Policy*: we won't do a job like "first have a model to predict price and then compose a rule based policy"
  - *Trained in a simulated environment*: We can add all the factors that would affect our model into the environment, and let the model learn to response to these factors (e.g. network latency)
  - *Learning to adopt to market conditions*
  - *Ability to model other agents*, this stems from the fact that we can incorporate other agent into our environment and let our agent to game with them

### [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
A good notebook introduce a basic workflow of how to solve a competition on kaggle. The most valuable part in my opinion is how to use pandas to do lots of data investigation and feature visualization.

### [Exploratory data analysis with Pandas](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68)
A good tutorial on explaining how to use pandas to do data analysis. The assignment is well-written and pretty helpful for mastering all kinds of functions in pandas.
* [Pandas Cheat Sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
