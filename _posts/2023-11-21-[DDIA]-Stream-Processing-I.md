---
layout: single
title: DDIA Chapter 11 Stream Processing Part I
tags:
- distributed system
- system design
- DDIA
- message queue
toc: true
---
In this post, we would introduce stream processing. Since it is a large topic, we would break it down into 2 part, and in the first part, we would focus on the component that is related to the "flow" of stream, a.k.a, **delivery of message**.

## What is Event
Stream is composed by sequence of *event*, which we also use *message* as an alternative term. Here is a quote from confluent on describing what is *event*

> An event is any type of action, incident, or change that's identified or recorded by software or applications. For example, a payment, a website click, or a temperature reading, along with a description of what happened. 

Take the payment as an example, a *payment event*, could be User A paid X dollars to User B, for the purchase of an item C, on date X. This event would be recognized by our system to trigger the necessary processing (e.g. record in database, make third-party API call).

## How to deliver message
How could we deliver message from machine A to machine B? There are multiple options.

### Direct connection
The most straight-forward approach is to build a direct connection between A and B via network. Once the connection is published, B could receive the message from A in *2 different patterns*
- Proactively asking A if there is new message with some intervals in between these ask
- Passively wait until A notify that there are some message for B to read

These 2 different patterns, more formally speaking, **poll** and **push**, is common approach on *how* message is delivered, or how *consumer* (B in our example) would receive the message.

Direct connection works, but what would happen if B somehow offline for a period of time $T$? B would miss all the message A plans to deliver during $T$. One potential solution is to add the capability of storing the message temporarily within A, but that would increase the responsibility of A and make it more complexity. We need some sort of dedicated component to help us, this lead to *message broker*, or *message queue*, which is really good at this job.

### Message Queue
Message queue could be treated as some type of *buffer* in between of the message sender, a.k.a producer, and message receiver, a.k.a consumer. Producer would publish message to message queue, message queue would do some "necessary" processing on the message and hold it. Consumer could retrieve these message from message queue, by subscribing to some queue. Since message is buffered in message queue, it is okay that B is offline when A tries to send message, message queue would hold that message, and when B comes online, the message is not lost and could be consumed.

### When to use Message Queue
Message queue is pretty good to be used when the business involves certain **async** property, which means that user don't expect an immediate response from the application, but could retrieve the result sometime in the future. Some typical case including:
- Job scheduler: user schedule a job (e.g. project building, model training) and expect it to finish sometime in the future
- Youtube video encoding: when user upload a video, the encoding job would be pushed onto a queue and be processed by some worker in the future
- Notification: a job to send some customer SMS/Email would be placed on queue and be sent in the future

In the later section, we would see some more concrete example from industry on how message queue is being used in practice.

Everything has two sides. The benefits of using message queue is that: **1. improve overall robustness of the system be decoupling different components**; **2. balance the workload for upstream/downstream system (e.g. in case of burst of traffic)**. The downside of message queue is that, it would increase the complexity of the overall system (e.g. how to handle duplicated events gracefully).

## Industry practice
### RabbitMQ & Kafka
[RabbitMQ](https://www.rabbitmq.com/) and [Kafka](https://kafka.apache.org/) is 2 commonly adopted message queue in industry. For a deeper dive into these 2 message queue, we would put it into another post. Here we would first summarize some highlight of them:

|   | RabbitMQ | Kafka |
|---|----------|-------| 
| **Message Persistent** | control by request parameter | persistent |
| **Message Delivery** | poll | push |
| **Message Ack** | auto-ack or explicit ack | no ack, consumer commit offset |
| **Scalability** | vertical | horizontal |
| **Availability** | single node in general | leader-follower replication |
| **Order Guarantee** | FIFO in general, special case: priority, sharded queue, multi consumer | FIFO on partition level |
| **Consumer Load Balance** | priority or round robin | different strategy specified by consumer group |
 
### DoorDash
In this [engineering blog](https://doordash.engineering/2020/09/03/eliminating-task-processing-outages-with-kafka/), DoorDash introduced how they are using message queue in their business and why they migrate from RabbitMQ to Kafka.
- Several business task in DoorDash is done in async, such as order checkout, merchant order transmission and dasher location processing
- DoorDash use Celery + RabbitMQ as their initial async task processing infra. However, they identified several pain points:
    - Availability is low. RabbitMQ would easily down during peak traffic. Traffic control needs to be enabled to prevent the issue that task consumption could not keep up with task publishing, which cause serious network lagging.
    - Scalability is low. They are running the largest RabbitMQ node already (vertical scale). And they are using the primary-secondary HA mode, which also prevent them from scale (the down time could easily goes to 20mins to recover)
- They migrate RabbitMQ to Kafka to achieve better availability (partition replicated) and scalability (partitioned topic)
    - They also mentioned on improvement on dealing with "straggler": using one dedicated thread to read message from topic partition, and use multi-threading to process the message. Thus, if one message takes long time to process, then only one thread would be blocked, while other thread could continues to process the messages

### Robinhood
In this [blog](https://newsroom.aboutrobinhood.com/part-i-scaling-robinhood-clearing-accounting/) from Robinhood, the author introduced how they are using Kafka to build their clearing service (which is one critical service to make sure the inside and outside account information is insync).
- Clearing service is not on the critical path of users (users don't need to be aware of this), and thus they decided to build it as an async service.
- In their initial design, they use a monolith consumer, which contains a giant transaction to make update to several tables. This raise the contention issue and the efficiency is low.
- In their new design, they breakdown the original transaction into several smaller transaction to update only 1 ~ 2 tables. They also adopt the event source pattern that, once one job is done (e.g. user table update finished), it would fire one event to a Kafka topic, and one downstream consumer would consume the event and to the necessary update (e.g. update account table), and then fire another event.
    - The benefit of this reduction in contention and overall throughput improvement
- But what if one consumer in the middle failed, how to resume and avoid duplicated write?
    - Use Kafka commit log to resume where left
    - When do the DB write, first update the lookup table, then the duplicated write would be no-op
