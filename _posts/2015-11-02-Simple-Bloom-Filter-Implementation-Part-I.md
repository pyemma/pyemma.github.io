---
layout: single
title: "Simple Bloom Filter Implementation Part I"
tags:
- project
- distributed system
- data structure
---

Recently, I'm studying some basic concepts in distributed system. The materials I'm using is [Distributed Systems Concepts](http://www.amazon.com/Distributed-Systems-Concepts-Design-5th/dp/0132143011/ref=sr_1_3?ie=UTF8&qid=1446522616&sr=8-3&keywords=distributed+system). I know that simply reading the book is far not enough: the concepts are abstract, but we need to handle partical problem. So I decided to do some simple projects, using some existing technoloy, to better understand these concepts and how to apply them to real world. These projects can be very very simple, and I'm definitely sure that there are better production available, but building something by your own hands can give you a feeling of achievement, no matter it is perfect or not, and this would be the biggest impluse for you to move on.

### Introduction
The first project is to implement a simple [bloom filter](https://en.wikipedia.org/wiki/Bloom_filter). Bloom filter is an advanced data structure mainly used for membership test (check if a given element is in or not). The advantage of adopting bloom filter is lower collision and lower space consumption. The traditional implementation of bloom filter is to use a bit vector. When we try to add a new element, we apply several different hash function on the element to generate several keys and use these keys as indexes to set the corresponding position in the bit vector to `true`. When we try to check if a given element is in it or not, we apply the same hash functions to generate a set of keys and we check if each position in the bit vector is `true` or not: only when all positions are marked as true, this element is considered to be in the set. The problem with bloom filter is that it can have false positive, but this kinds of error is relatively small. Typacilly, I support two kinds of operations: `add` and `contain`. `add` is used for adding an element and `contain` is used for checking existence.

Currently, there is still nothing related to **distributed system**. What it happens is here: I make this bloom filter a distributed service. We can run a bloom filter service on a machine, and call the functions `add` and `contain` from another machine, or we can run the service in one process and call the functions from another process within the same machine. In both side, the caller could not directly access the content in bloom filter, and thus the functions have to be remotely invoked. I use **Thrift** to implement this part. Thrift is a set of software stack that help implement cross language PRC (remote procedure call): it contains IDL (interface definition language) to help define data structure to be used and the interface of the service; it can also generate necessary code such as mapping data type to a specific language's supported data type accroding to your configuration file.

### Implementation details
The project can be found [here](https://github.com/pyemma/BloomFilter). Now, I will explain some details of the implementation.

The most impotant thing in `BloomFilter<T>` class is that I use a list of `Hashable<T>` object to represent the hash functions. `Hashable<T>` is an interface with only one function `hash(T t)`, which takes one type of element as input and return an integer. We can design different kinds of concert class to support different kinds of hash function for different types, e.g. I create a StringHash class which can hash a string data. We can add different kinds of `Hashable` on the same type `T` to this list. This kinds of design is called **Strategy Pattern**. It utilizes composition to reduce the cohesion of the code.
{% highlight java %}
public class BloomFilter<T> {
    private List<Hashable<T>> hashFunctions;
}

public interface Hashable<T> {
    // return a integer by applying some method on the object
    public int hash(T t);
}
{% endhighlight %}

The definition of the service is quite simple, it is only some simple Thrift statement.
{% highlight text %}
service BloomFilterService
{
    void add(1: string str);
    bool contain(1: string str);
}
{% endhighlight %}

The following part is modified from the online tutorial [here](http://thrift-tutorial.readthedocs.org/en/latest/usage-example.html).
After run thrift with command `thrift -r -gen java bloomfilterservice.thrift`, it will create a class called BloomFilterService.class in a folder called gen-java. It contains all necessary code for implementing RPC. The things we need to do is:
1. Implement the core logical to actually provide the service
2. Implement server code to run the service
3. Implement client code to call the service

I create a class called BloomFilterHandler.java to handle the actual logic of bloom filter service. In this handler, it contains a `BloomFilter` object as its data member and it implements an interface provided in BloomFilterService.java, called `BloomFilterService.Iface`. This interface contains the provided to API can be called by clients. An instance would be passed to a processor introduced later to provide the service logic.
{% highlight java %}
public class BloomFilterHandler implements BloomFilterService.Iface {

    private BloomFilter<String> bf;

    public BloomFilterHandler() {
        bf = new BloomFilter<String>(1000);
    }

    public BloomFilterHandler(List<Hashable<String>> functions, int size) {
        bf = new BloomFilter<String>(functions, size);
    }

    public void add(String str) throws TException {
        System.out.println("Operation: add " + str);
        bf.add(str);
    }

    public boolean contain(String str) throws TException {
        System.out.println("Operation: contain " + str);
        return bf.contain(str);
    }
}
{% endhighlight %}

The next thing is the server code. In the server code, I create a list of `Hashable` object to be passed into the bloom filter in the `BloomFilterHandler`. Then we create a `processor` also defined in `BloomFilterService` and use the `handler` we created to initialize it. The rule or `processor` is to read in the parameters, call the `handler` provided and then return the output. We create a `server` provided by Thrift and use a `transport` object to initialize the server. The `server` would response for dispatch the call to corresponding function and the `transport` object would response for read and write to wire.
{% highlight java %}
public class BloomFilterServer {

    public static BloomFilterService.Processor processor;

    public static void main(String[] args) {
        try {
            List<Hashable<String>> functions = new ArrayList<Hashable<String>>();
            functions.add(new StringHash(17));
            functions.add(new StringHash(23));
            functions.add(new StringHash(31));
            BloomFilterHandler handler = new BloomFilterHandler(functions, 1000);
            processor = new BloomFilterService.Processor(handler);

            Runnable simple = new Runnable() {
                public void run() {
                    simple(processor);
                }
            };

            new Thread(simple).start();

        } catch (Exception x) {
            x.printStackTrace();
        }
    }

    public static void simple(BloomFilterService.Processor processor) {
        try {
            TServerTransport serverTransport = new TServerSocket(9090);
              TServer server = new TSimpleServer(new Args(serverTransport).processor(processor));

              System.out.println("Starting the simple server...");
              server.serve();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }    
}
{% endhighlight %}

The client code is relative simple. We also use an object provided in the `BloomFilterService` called `BloomFilterService.Client`. We create a such object by providing a `transport` object, determining how data would be written onto wire, and a `protocol` object, determining how data would be serialized and deserialzied. Then use the `client` object, we can call the API provided by the service.
{% highlight java %}
public class BloomFilterClient {
    public static void main(String [] args) {
        try {
            TTransport transport;

            transport = new TSocket("localhost", 9090);
            transport.open();

            TProtocol protocol = new TBinaryProtocol(transport);
            BloomFilterService.Client client = new BloomFilterService.Client(protocol);

            perform(client);

            transport.close();
        } catch (TException x) {
          x.printStackTrace();
        }
    }

    private static void perform(BloomFilterService.Client client) throws TException {
        client.add("apple");
        client.add("banana");
        System.out.println("Is apple there? " + client.contain("apple"));
        System.out.println("Is pineapple there? " + client.contain("pineapple"));
    }
}
{% endhighlight %}

### Notes
There are typically five components in RPC: client stub procedure, communication module, dispatcher, server stub procedure and service procedure. The mapping of Thrift components to these five components can be viewed as:
1. client stub procedure => BloomFilterService.Client
2. communication module => server and transport object
3. dispatcher => server object
4. server stub procedure => BloomFilterService.Processor
5. service procedure => BloomFilterHandler

### Next step
The current version bloom filter only use a very simple data type `String`. The next is to use a more complex data type to test if it works or
