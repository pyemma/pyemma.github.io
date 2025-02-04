---
title: How to Design Slack
date: 2025-02-01
author: pyemma
categories: [Distributed System]
tags: [distributed system, stateful service, websocket]
toc: true
math: true
---

In this post, I would like to discuss how to build slack, a very popular realtime messaging application, especially for group messaging (a.k.a channel) in cooperation messaging scenario. In this post, I will focus on discussing some key functionalities and have some light discussion on other functionalities that is not covered in this post to the last section of this post.

## Functional Requirements

1. Support channel message (a.k.a group messaging) and thread message (a.k.a reply to a message in a channel or DM)
2. Support direct messaging (a.k.a 1 to 1 messaging), support sending message both online and offline

## Non Functional Requirements

1. High availability
2. High scalability
3. Low latency
4. Causal Consistency

## Assumption

1. In this design, we would leave out the discussion how user would login and how user would join a channel. We assume that there is dedicated service to handle the user login verification, and user could join a channel via different approach such as invitation or search.
2. We would also simplify our discussion to persist user's channel membership information. In reality, these information could be stored in a dedicated `channel_membership` table. Based on the read/write ratio, relational database such as MySQL or column database such as Cassandra could be adopted.
3. We would assume that the message would be persistent on the server side as well, instead of only storing them locally on client side. This is a legit use case for the cooperation scenario, but might not be an option for other scenarios. For example, Wechat and Whatsapp does not store the message on the backend side but just temporarily buffer the message. Once the receiver is online and the message is delivered, the message on server side would be deleted.

## Key Challenges

1. How does client communicate with backend service
2. When one user send one message in a channel, how this message gets fan-out to other users in the same channel
3. How to guarantee the casual consistency within the channel messaging

## High Level Design

Here is a high level design of Slack, there are some alternative options with different trade off for some components, and we would discuss them shortly.

![Slack - Dispatcher Sytle](/assets/slack-dispatcher.png)

### Client connection

Client (e.g. desktop client or mobile app) connects with our backend service via **websocket** connection, which is handled by the `Websocket Server`. There are different approach for long live connection as well (short live connection is not very appropriate in this scenario due to large overhead on repeating building connections), e.g. *HTTP long polling*, *SSE*. However, realtime messaging is usually **bidirectional communication** and **websocket** would be good option in this scenario. Within each `Websocket Server`, there would be an event loop that is listening on the port it exposed, as well as an internal data structure to manage the channel and websocket connection relationship (a.k.a a subscription table, see example below). When `Websocket Server` received message for some channel (we would how this message is delivered here in next section), the event loop would check this data structure to find the websocket connection that is subscribing information from this channel. And the message would handler to the websocket connection to delivery to client.

```python
{
    'gongzuoqun_1': [websocket_obj1, websocket_obj2],
    'gongzuoqun_2': [websocket_obj1, websocket_obj2, websocket_obj3],
}
```

After the user has login, the user would build connection with one `Websocket Server` and the selection of the server is handled by the load balancer. There could be some different strategy, such as round robin, routing based on the workload on the server, or even provide sticky routing if it is just short time disconnection from the client. Once the connection is built, there are 2 important jobs that need to be done:

1. From the `channel` table to retrieve the channel information of the user, update its internal data structure, and send subscribe request to the `Channel Server` (to be expanded later)
2. From the `message` table to retrieve the message sent by others during the user is offline  

### Message Delivery

Within the topic of message delivery, there are 3 aspects that we need to discuss:

1. How does user send message to the channel
2. How does other users receive the message that are sent to the channel
3. How does user receive message when the user has been offline for a while

#### How does user send message

#### How does other users receive message

#### What does the data flow look like for users just become online

## High Availability

## High Scalability

## Additional Functionalities

## Reference

1. [从0到1：微信后台系统的演进之路](https://cloud.tencent.com/developer/article/2232218)
2. [Real-time Messaging](https://slack.engineering/real-time-messaging/)
3. [Redis Pub/Sub In-Depth](https://medium.com/@joudwawad/redis-pub-sub-in-depth-d2c6f4334826)
4. [How Discord Stores Billions of Messages](https://discord.com/blog/how-discord-stores-billions-of-messages)