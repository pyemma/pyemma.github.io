---
layout: single
title: "Simple Bloom Filter Implementation Part 2"
tags:
- project
- distributed system
- data structure
---

### Introduction
In the last [blog](http://pyemma.github.io/notes/Simple-Bloom-Filter-Implementation-Part-I/), we introduced the initial version of bloom filter. In the first implementation, we only tested our bloom filter on built in type `String`. This time, we tested in against a custom class `Person`. The idea is simple: we use thrift to define our custom data structure. Thrift will help us create a corresponding `Person` class. Then we define our own `Hashable<Person>` to be passed into our bloom filter.

### Implementation details
Here is our new thrift file. Notice that we need to assign identifier to each data field, since they would be used for serialize de deserialize data instead of the actual variable name(space consideration).
{% highlight text %}
struct Person
{
    1: string firstName,
    2: string lastName,
    3: i32 age,
    4: string email
}

service BloomFilterService
{
    // void add(1: string str);
    // bool contain(1: string str);
    void add(1: Person person);
    bool contain(1: Person person);
}
{% endhighlight %}
If we run the command `thrift -r -gen java bloomfilterservice.thrift`, the content in `gen-java` would also contain a generated `Person` class. We need to implement a `Hashable<Person>` type to hash a `Person` object into several integers. Here, the method I used is very simple, it still take a prime and the result would be a combination of all data fields.
{% highlight java %}
public class PersonHash implements Hashable<Person> {

    private int prime;

    public PersonHash(int prime) {
        this.prime = prime;
    }

    public int hash(Person person) {
        int sum = 0;
        String firstName = person.firstName;
        for (int i = 0; i < firstName.length(); ++i) {
            sum  = sum * prime + firstName.charAt(i);
        }
        String lastName = person.lastName;
        for (int i = 0; i < lastName.length(); ++i) {
            sum = sum * prime + lastName.charAt(i);
        }
        String email = person.email;
        for (int i = 0; i < email.length(); ++i) {
            sum = sum * prime + email.charAt(i);
        }
        sum += prime * prime * prime * person.age;
        return sum;
    }
}
{% endhighlight %}
The remaining is all every simple, we passed in a list of `PersonHash` object into handler and replace `String` with `Person` at correct places. Then all is done.

### Notes
* When I was trying to implement this part, I was planed to create a *general* bloom filter, that is utilize Java's template feature. However, I failed. Since I don't know how to define an API with template type in thrift.
* The package in Java is quite important and I have always ignore it before. During the implementation, I initially put `PersonHash` under `bloomfilter` package. This class need to access `Person` class generated by thrift, however, `Person` class is under the root package and `PersonHash` could not access it. I was confused by this problem for a long time.

### Next step
* Try if thrift support function overload. Have function for `String` and `Person` at the same time to see if it still works.
* Currently, I only tried it with toy data. I decided to move on to some real data to test the performance of bloom filter.
* Add some performance measure code.
