---
layout: single
title: "Clean Code Summarize"
---
Here is just some tips I think is most useful for guiding me to write better code from book `Clean Code: A Handbook of Agile Software Craftsmanship`.

##Naming
We cannot avoid naming things well we are writing code, a good name can considerably increase the readability of the code. We even does not need to write comment in code since the code itself has already been revealing itself and everyone can easily get what you want to do just from the name you given to some variables.

1. Use intention revealing names, this can be some single words or some combination of words. Some may argue that using intention revealing names can lead to a lonnnnnnnnng name, but I think with some proper abbreviation, this is not a big problem.
2. Avoid use wrong words which provide wrong information. Use pronounceable names, which is helpful during discussion, and searchable names, which is helpful during debugging and refactoring.
3. Class and objects should have noun or noun phrase names, methods should have verb or verb phrase names.
4. Keep one world per concept, `get`, `fetch`, `retrieve` all referring to take something out, but keep use only one words throughout your code and keep consistence.

##Function
We will write a lot functions to help us fulfill all kinds of tasks. Write good functions can better reveal our intention of the code and make it easier to maintain.

1. The most important thing is that functions should be small and each function should only do one thing and do it best. How to measure what is a one thing is hard and I feel it depends on your experience a lot. According to the author of the book, he suggest use a to do sentence to describe one thing. Another way to check is to see if you can extract another function from it with a name that is not merely a restatement of its implementation.
2. Read code in a top down matter, put the called function just under the function calls it.
3. Avoid using switch statement in function, use polymorphism to replace it. Only use statement in object creation process(object factory) where it is not avoidable.
4. Be mean to your function parameters, function with no parameter is best, then comes one parameter and two parameters. Functions with three or more parameters should be avoided(considering put them into one argument object). And don't use boolean variable as your parameter......
5. If a function has side effect(create new objects, insert new record), this should be showed in its name. Each function's responsibility should either be querying something or processing something, but should avoid doing these two things at the same time.
6. Prefer raise exception to return error code.
7. Don't Repeat Yourself (DRY). Remove as many duplicates as possible. In my experience, I usually do this step in refactoring step.

##Objects and Data Structures
Data structures expose data and no meaningful functions. Objects hide data and provide operation on data.
Procedural code makes it hard to add new data structures because all the functions must change. OO code makes it hard to add new functions because all the classes must change.
`Law of Demeter` says that a method `f` of a class `C` should only call the methods of these: `C`, an object created by `f`, an object passed as an argument of `f`, an object held in an instance variable of `C`.

##Boundaries
In our code, we might use third party library, this kinds of code create a boundary in our system. We need to protect our code to not effected when the third party library has changed. Use some design pattern like adapter can avoid these kinds of bad effect.
When learning third party library, writing some simple test to explore the understanding of the third party code. In latter, these simple test can be used as boundary test in our system.

##Unit Test
I think this part gives me most improvement over code development. In the past, I seldom write test for my code, and it is really painful when I need to change my code to check to see if my logic is still right or not. Right now, I would first define a set of test first and then start to write the actual logic code. Latter, I will not be afraid to refactor and update my code, since I have ground truth now to check my code is right or not.

1. Write test first, and the test code should cover as many cases as possible.
2. Each test should only test only one concept or case, don't mess many cases in one single test.
3. Test should run fast, independent and repeatable. They should only have two value, pass or not. And always write test in a timely fashion.

##Class
Like function, Class should also be small. A class or module should have one and only one responsibility.
Cohesion can be defined as private variables utility rate. All functions in the class should use all private variables in the class. Cohesion should be as high as possible. If some variables are only used by a small portion of code, this is a suggestion that we can extract another class and encapsulate these functions.
Class should be open for extension but close for modification. Adding feature by extending instead of modifying origin class. (This is hard in practical and require a lot of experience, some good design pattern can be helpful)

##Emergence
1. Runs all tests
2. Contains no duplications
3. Express the intent of the programmer
4. Minimizes the number of classes and methods

##Concurrency
I have no experience in terms of writing concurrency code. I think I will learn this part recently and have another blog to summarize some basic concepts.
