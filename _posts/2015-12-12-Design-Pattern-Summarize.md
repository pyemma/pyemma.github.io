# Design Pattern Summarization

**Strategy Pattern**
Defines a family of algorithms, encapsulate each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
[Image: file:///-/blob/eDDAAAJA4tZ/2DMfZKq387ZdSBlQHdKHXg]Use composition instead of inheritance to separate the detailed actions from the clients code. The detailed actions can be determined or changed in run time instead of before compile. In this way, when we want to change or add some behaviors, we only need to change the detailed class that response for this kind of behavior instead of modifying the client code. Besides, the class response for different actions are also reusable since they are only responsible for behavior, which is a kind of abstract concept. Several client codes might share same behavior.

**Observer Pattern**
Defines a one-to-many dependency between objects so that when one object changes state, all of its dependents are notified and updated automatically.
[Image: file:///-/blob/eDDAAAJA4tZ/e5yAyO3CQ1Zy4FAFFPAWeg]The subjects interface defines a observable (which can be observed), it defines how to add/remove/notify observers. Observer interface only defines an update() method. The subject object will hold a list of observer object, and when something changed, it will call each observer objects’ update() function. The update() function is override by the concrete observer object.

1. Observers are loosely coupled with observable since observable knows nothing about them. The only thing observable knows is that it can call update function to inform observers.
2. Observers can push or pull data from observables.
3. Don’t depend on the order of the notification.

**Decorator Pattern**
Attaches additional responsibilities to an object dynamically.  Decorators provides a flexible alternative to subclassing for extending functionality.
[Image: file:///-/blob/eDDAAAJA4tZ/CZDVKsQpkr5esaaJQoFwgg]Java I/O use decorator pattern to add additional functionality.
[Image: file:///-/blob/eDDAAAJA4tZ/SIGPLcv8wUOvYcMxmfma8Q]
