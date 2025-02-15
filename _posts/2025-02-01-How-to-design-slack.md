---
title: How to Design Slack
date: 2025-02-01
author: pyemma
categories: [Distributed System]
tags: [distributed system, stateful service, websocket]
toc: true
math: true
---

In this post, I would like to discuss how to build slack, a very popular realtime messaging application, especially for group messaging (a.k.a channel) in cooperation messaging scenario.

## Functional Requirements

1. Channel messaging (group messaging) and thread message (reply to a message in a channel or DM)
2. Direct messaging (1 to 1 messaging)
3. Support sending message to both online and offline users

## Non Functional Requirements

1. High availability
2. High scalability
3. Low latency for realtime message delivery
4. Causal consistency for best user experience
5. At least once delivery semantic to avoid missing messages

## Assumption

1. In this design, we would leave out the discussion how user would login and how user would join a channel. We assume that there is dedicated service to handle the user login verification, and user could join a channel via different approach such as invitation or search.
2. We would also simplify our discussion to persist user's channel membership information. In reality, these information could be stored in a dedicated `channel_membership` table. Based on the read/write ratio, relational database such as MySQL or column database such as Cassandra could be adopted.
3. We would assume that the message would be persistent on the server side as well, instead of only storing them locally on client side. This is a legit use case for the cooperation scenario, but might not be an option for other scenarios. For example, WeChat and Whatsapp does not store the message on the backend side but just temporarily buffer them. Once the receiver is online and the message is delivered, the message on server side would be deleted.

## Key Challenges

1. How does clients communicate with backend services
2. When one user send a message in a channel, how this message gets fan-out to other users in the same channel
3. How to guarantee the casual consistency within the channel messaging

## High Level Design

Here is a high level design of Slack where we adopted a hierarchy broadcasting design. There are some alternative options with different trade off and we would discuss them shortly in the sections below.

![Slack - Dispatcher Style](/assets/slack-dispatcher.png)

### Client Connection

Client (e.g. desktop client or mobile app) connects with our backend service via **websocket** connection, which is handled by the `Websocket Server`. There are different approaches for long live connection as well (short live connection is not very appropriate in this scenario due to large overhead on repeating building connections), e.g. *HTTP long polling*, *SSE*. However, realtime messaging is usually **bidirectional communication** and **websocket** would be good option in this scenario. Within each `Websocket Server`, there would be an event loop that is listening on the port it exposed, as well as an internal data structure to manage the channel and websocket connection relationship (a.k.a a subscription table, see example below). When `Websocket Server` receive message for a channel, the event loop would check this data structure to find the websocket connections that are subscribing this channel and the message would be handled to the websocket connection to delivery to the client.

```python
{
    # channel_id: list[websocket_objects]
    'gongzuoqun_1': [websocket_obj1, websocket_obj2],
    'gongzuoqun_2': [websocket_obj1, websocket_obj2, websocket_obj3],
}
```

After the user has login, the user would build connection with one `Websocket Server` and the selection of the server is handled by the *load balancer*. There could be some different strategy, such as round-robin routing, workload based routing, or even sticky routing if it is just a short-time disconnection. Once the connection is built, there are 2 important tasks that need to be done:

1. Retrieve the channel information of the user from the `channel_table`, update internal subscription table, and send subscribe request to the `Channel Server` (to be expanded later)
2. Retrieve the message sent by others during the user is offline from the `message_table`

When user is offline, the `Websocket Server` would also update its internal subscription table. If for some channels there are no clients subscribing, the server could send a unsubscribe request to the `Channel Server` so that we could manage the workload and avoid unnecessary resource waste.

### Message Delivery

Let's first discuss the option to deliver channel message. There are 3 aspects that we need to discuss:

1. How does user send message to the channel
2. How does other users receive the message that are sent to the channel
3. How does user receive message when the user has been offline for a while

### How does user send message

To send a message to a channel, there are two approaches: 1. use the websocket connection we have already established; 2. send it as a HTTP Post request to another `Web Server` (a.k.a functional sharding). Both of the choice are workable solution (Slack used the 2nd approach according to their [blog](https://slack.engineering/real-time-messaging/)). Here is a quick comparison between the 2 options:

- The benefit of reusing websocket connection is minimized latency and reduced complexity on the infrastructure; however, the downside is that the websocket server needs to keep live connections with clients and would not be easily to horizontally scale based on the QPS of the traffic; also websocket requires customized retry logic when message sending is failed
- The benefits of using HTTP Post request to send message is that it offers better horizontal scalability and native retry support; however, the downside is that the latency would be slightly higher and there is some more complexity on the infra

Another practical consideration is the processing of the message. To make sure that we don't lose any message, when our backend receive the message, we would first write it into the `Message DB` and then fan-out (or broadcast) the message to other users in the same channel. There would be additional business related logic need to run over the message such as validation. All of these process would put additional pressure on the host machine. Thus, it might be a legit option to have dedicated `Web Servers` to handle message write instead of reusing the `Websocket Server` to avoid putting too much load on them.

#### How does other users receive message

Let's first take a look at the channel messaging scenario. There are several options:

- *Dispatcher*: In this option, we would have dedicated `Channel Servers` which play as a *dispatcher role* (I talked about this design in [my pervious post on how to design auction service]({{ site.baseurl }}{% post_url 2024-04-06-How-to-design-auction-system %})). Each channel service would handle a portion of channels (we will discuss this in scalability section) and maintains a subscription table similar to one in `Websocket Server`  which records the `channel_id -> websocket server id` information. This subscription table is updated by the request sent from `Websocket Servers`. Once the web server receive messages, it would send this message to the channel server the manage the channel and then the message would be broadcast the websocket servers that is subscribing to the channel.

> You might ask why we need such a `Channel Server`, why not just use the `Websocket Server`? One reason is that the subscription table, which is a stateful information, is not easy to be directly maintained on `Websocket Server`; another one is to follow the *Single Responsibility Principle*; last but not the least, is that we could scale `Channel Server` and `Websocket Server` individually based on the number of channels and number of active users

- *Pub/Sub*: In this option, we would use message queue for pub/sub style message exchanging. For each channel, a dedicated topic would be created within the message queue. For the `Websocket Server` who is connected with the user in the channel, they would subscribe to the corresponding topic. Once message is published onto the topic, message queue would deliver the message to the `Websocket Server` who is subscribing. Note that we could choose a message queue that support poll or push mode: for poll mode, `Websocket Server` needs to periodically poll data from the queue; for push mode, message queue would be responsible for delivering message. Given the load and the realtime scenarios, adopting a **in memory push mode message queue** might be a good choice.

- *Fan-out* In this option, we maintain a queue similar to a inbox for each user, and the channel message would be fan-out written to this inbox. The exact choice of the inbox could be message queue (each user's inbox is a topic) or database (each user's inbox is a table). `Websocket Server` would periodically poll data from the inbox for new message. One benefits of this solution is a unified flow for both online and offline message delivery scenario. However, this solution would probably incur higher latency, as well as the causal inconsistency issue (because we have multiple copy of message stored for the same channel).

We also need to consider how could be provide the *at least once* message delivery guarantee as we mentioned in the non functional requirements.

- When client send a message to a channel, if the HTTP post request failed, client side could retry it. With retry, there could be duplicated requests sent from client. We could adopt an *idempotency key* and *cache layer* so that the web server could dedup the already succeed post requests. In our design, each message needs to be written to `Message DB` first for persistent. Thus we could also use `upsert` when writing to DB to avoid writing duplicated message with the same *idempotency key*. This strategy is also applicable to the *Fan-out* option.
- In the *Dispatcher* or *Pub/Sub* option, when `Web Server` forward messages to `Channel Server` and `Channel Server` broadcast messages to `Websocket Server`, we could also adopt retry until we receive ACK from the destination server. We could add a dedup on the client side when receive message from the websocket and allow `Web Server` or `Channel Server` to forward duplicated requests.

In the remaining design, we would assume to use the *Dispatcher* option, which is also the option that Slack has deployed in production.

#### How to receive offline message

If a user is offline, then he would miss the message that is broadcast by the `Channel Server`. To receive these messages, we need to leverage the `Message DB`. When a user is offline, we could store the timestamp as `last_active_time` in a dedicated table (e.g. `user_activity_table` or in `user_profile_table`); and when user is back online, we could retrieve the `last_active_time` and do a query in `Message DB` against all channel that user is in, pulling out all messages. Another option is to keep a client side snapshot of the local message status, and send it to server when user come back again; server would use the snapshot to compute the delta message the user is missing and send it back. Both of the solution has some drawback: the `last_active_time` solution requires additional DB write and also need the time within the fleet of machine is synchronized (e.g. NTP) or read more data backward; using snapshot would have some additional challenge when dealing with multi-device scenario.

One more thing to mention is that we need to appropriately synchronize offline message receiving and online message receiving when user is just back online. We should confirm that all the offline message is read and shown to user before showing the message from websocket in order to guarantee the causal consistency and good user experience.

#### How to support direct messaging

Direct messaging and channel messaging does not have too much difference, except that channel messaging is broadcasting to multiple users while direct messaging is "broadcasting" to only one users. However, it might be less efficient to let the `Channel Server` to forward the message. We could directly forward the message to the `Websocket Server` that the receipt user is connected. Which `Websocket Server` the user is connected to could be store in a in memory **KV store** for fast read/write. Another consideration is to treat it as *service discovery* and use some dedicated solution such as **Zookeeper** or **Consul**; but since user would online/offline pretty often, this could put a lot of write pressure for the distributed consensus.

#### How to support thread message

Thread message does not have a big difference compared with regular message. The only difference is that they are embedded within another message in channel. To support it, we could create a field in the `message_table` to indicate which message is its parent. And when we broadcast the message, we also include this information in the request payload.

```python
message_table
id: int
channel_id: int
author_id: int
parent_message_id: Optional[int]
content: text
timestamp: int

# message payload
{
    "content": "abc", "parent_message_id": 123, "id": 1234
}
```

## High Availability

Let's discuss about the high availability. We would skip the discussion on `Message DB` as the general rule of how to make DB fault tolerance is applicable (you could checkout my Youtube video [here](https://www.youtube.com/watch?v=X8IhZx7fg24)).

For the `Websocket Server`, they are stateless and don't need to persistent any data on the server. If one `Websocket Server` is dead, then all connections on that server would be lost. The clients need to reconnect with other `Websocket Server`. The new `Websocket Server` would also need to refresh the information stored in the **KV store** (could use last write win strategy) to make sure direct messaging could still work.

For the `Channel Server`, they are stateful as they need to maintain the subscription table so that channel message could be correctly broadcasted. First, this subscription table could be maintained locally on the host disk using WAL. If the process that responsible for forwarding the message crashes accidentally but the host is still alive, we could rebuild the subscription table quickly by reading the WAL. In the situation that the host is gone, we need to replicate the subscription table to somewhere else to make sure the message could still be broadcast. One option would be storing this information to KV store; another option is to directly replicate the subscription table to other `Channel Server`. In this option, we could have a single leader to handle all read/write request, and have multiple followers to store the replicated data. We also need a coordination service to help maintain the leader/follower information as well as the leader election.

![Channel Server Replication](/assets/channel-server-replication.png){: width="500" height="500" }

## High Scalability

The `Message DB` could be horizontally scaled. For sharding, there is key-range sharding and hash based sharding. Key-range sharding is not very suitable in this case. We could apply hash based sharing on the channel id, which should be able to relatively evenly distribute the message across all hosts. To support efficient query data for offline case, we could build index on timestamp column (since each channel's message is co-locate on the same host, local secondary index is sufficient). Another consideration is to partition based on both channel id and the bucketed timestamp similar to [how Discord store their data initially](https://discord.com/blog/how-discord-stores-billions-of-messages).

The `Websocket Server` is relative simple to scale because they are stateless, we could add more servers if the network bandwidth is limited. The `Web Server` that handles the HTTP post requests for sending message could also be handled in the similar strategy.

The `Channel Server` could be scaled by horizontal scaling as well, to partition the `channel_id` and have each server to handle a portion of the channels. However, since we are dealing with realtime messaging scenario, it is pretty critical to minimize the latency during the time that one node is down or we add additional node into the cluster to handle more traffic. That is to say, minimizing the data we need to transfer during rebalancing is critical. Thus, adopting **Consistent Hashing** would be a great option here. Which server handles which range of ids on the *consistent ring* is a consensus that needs to be made among all servers. This could be achieved by running a *gossip protocol* among all servers or use a dedicated coordination service to manage this assignment information. The downsides of *gossip protocol* is that there would be some performance penalty when the number of nodes in the cluster is large; the cons of using coordination service is that we need to manage and maintain another service within our system.

> To figure out which `Channel Server` to send the message to, `Web Server` could first check the coordination service or any channel server to get the assignment information and cache it locally to reduce the volume of requests. Since this information is not changed that frequently, we could refresh this information on a cadence or use read-repair strategy (refresh the info when channel sever refused the request due to staled information)
{: .prompt-info }

Although we use **Consistent Hashing** and `channel_id` to balance the data, there could still be unbalanced load on each server, especially when the number of physical node is relative small. We could consider using **virtual node** to make the data on each node more balanced; or consider manual configuration for some hotspot channel.

## Causal Consistency

In group messaging scenario, it is pretty critical to guarantee causal consistency; otherwise users in the group might see wired conversation. In our design, we could guarantee this with a single leader replication strategy. Since the message writing is first written to `Message DB` and then broadcast to users, and we shard the `message_table` based on `channel_id`, we could guarantee that the order of the message has and only has a single "version" and there would be no conflict. If we make db write and message broadcast async, then there would be teh case that the order we written to DB is not the same with the order we broadcast, and the causal consistency is break. One potential solution to support async DB write and message broadcasting is to associate a version vector to capture the causality among messages.

![Causal Consistency Issue In Async DB Write Mode](/assets/causal-consistency-issue.png)

## Additional Functionalities

We have finished the discussion on the majority part of the functionality, let's have some light discussion on other functionality but going to deep.

- How to support online presence feature
  - There are different options to implement this, in push mode or in poll mode. In the push mode, we could have a dedicated queue for each user to receive the online signal. When user is online, the client could send a active message periodically via the websocket and forward this message to all users' friend (or is viewable in client); when user is offline, we could also send such a message to the queue to indicate that the user is offline. This online signal is similar to heartbeat that is widely used in distributed system. In poll mode, the `Websocket Server` could poll information from the coordination service where we store the `user id <-> websocket server id` info and use that to indicate if user is online.
- How to support *XXX is typing...**
  - We could reuse the websocket and send a typing message to the `Websocket Server`, and then have this message to be broadcast by the `Channel Server`. Since this type of message does not need persistent, we don't need to go through the `Web Server` for simplicity.
- How to support global service
  - To support global service, we could deploy the `Websocket Server` to each global region and have user connect to the region that is closet to them. The `Channel Server` could be in a dedicated region to have the single leader setup. But we could also consider multi leader setup to further scale, with the drawback of write conflicts in some cases.
- How to support Slack emoji
  - Emoji could be treated as a special type of thread message. We could reuse the data schema defined for thread message, by adding a `message_type` field with enum type such as `text`, `emoji`.

> If you find this post helpful, feel free to scan the QR code below to support me and treat me to a cup of coffee
{: .prompt-tip }

![Thank You](/assets/qr%20code.png){: width="300" height="300" }

## Reference

1. [从0到1：微信后台系统的演进之路](https://cloud.tencent.com/developer/article/2232218)
2. [Real-time Messaging](https://slack.engineering/real-time-messaging/)
3. [Redis Pub/Sub In-Depth](https://medium.com/@joudwawad/redis-pub-sub-in-depth-d2c6f4334826)
4. [How Discord Stores Billions of Messages](https://discord.com/blog/how-discord-stores-billions-of-messages)
5. [Cassandra - Dynamo - Virtual Node](https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html)