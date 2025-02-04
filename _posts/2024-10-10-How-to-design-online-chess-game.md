---
title: How to Design Online Chess Game
date: 2024-10-10
author: pyemma
categories: [Distributed System]
tags: [distributed system, stateful service, message queue]
toc: true
math: true
---

I have been a fan of online game since my parent bought me a PC when I was in 5th grade. I could still feel the excitement when every Thursday night I rushed home and played DaHuaXiYou 2 with my friends. This is also the direct reason why I decided to major in computer science in college. Today, let's discuss something related to online game: how to design an online chess game.

## Functional Requirements

Here is a minimal set of functional requirements for our online chess game:

1. Two users could play chess against each other, 1-1 game, take move by turns
2. Player could revert their last move if their opponent hasn't made a move yet
3. Player could resume from temporary disconnection and join back the same game

## Non Functional Requirements

1. High availability, no single point of failure
2. Low latency
3. High scalability
4. Consistency, both player should see the same board state

## Assumptions

These are some basic assumptions we would made in our design. In real world system design interview, these would be some good questions to get clarification from the interviewer:

1. In this discussion, we would only focus on the core logic of the game. Additional features, e.g. player ranking, profile system and player matching, would be out of the scope for today's post and I plan to have a separate post to discuss them
2. We also assume that each player could be at most one game at a time, this would help us to simplify our discussion. However, it should be relative straightforward to integrate `session` management

## Backend Envelope Estimation

We would assume that our online chess game is pretty popular and have a DAU of 100 million based on this [source](https://en.wikipedia.org/wiki/Chess.com#:~:text=Chess.com%20said%20it%20reached,users%20as%20of%20April%202023). And since its a chess game, user need to take turns to make a move, we could assume that on average each player takes 10 second to make a move. We assume each player would play 10 games every day. This translate to about $100 million / 10^5 * 10 (games / day) * 0.1 (move / sec) = 1000 QPS$. We could assume that at peak time, the QPS could be 5x which gives us 5000 QPS.

On the storage part, we actually don't need to have too much storage (which we would discuss more in details in the next section), as either the game state nor the moves take much space to store. For the visualization element such as the picture of the board, they could be shared across all games and is some statistic resource, which could be stored in CDN. We assume that each game would require 10KB storage, which gives us about 1TB storage per day.

## Key Challenges

Here is some technical challenges on top of my mind, we would discuss how different design choices going to affect the answer of these questions:

1. How to maintain the state of the game?
2. How to keep users connected with each other to paly the game?
3. How to handle difference racing conditions? (e.g. Player A make a move while Player B tries to revert move)
4. How to handle different failure mode? (e.g. Player A is disconnected due to network issue)

## High Level Design

In this post, we would introduce 2 different architectures. There are 2 different philosophies on how to represent game state: keeping a latest view of the game, which is pretty common practice in database community; or keeping a series of event that has happened, a.k.a event sourcing, which is a relative new practice in the world of distributed system. These 2 different philosophies lead to 2 different design choices: **stateful game server** and **pub/sub based architecture**.

### Stateful Game Server

A high level design of this approach is shown below:

![Stateful Game Server](/assets/online-chess-game-server.png){: width="700" height="400" }

The `Game Server` component would be responsible for maintaining the game state and connection with players. The game state is relative simple, it could some something like this:

```python
# game state
board: str  # A 8x8 board represented as a string, different characters represent different pieces, e.g. 'Q' for queen
current_turn: bool # True if it's player 1's turn, False if it's player 2's turn
is_win: bool # True if the game is won, False otherwise
```

The game state is not large and we could maintain it in memory together with other game metadata such as `game_id`, `player_id` for the lowest latency. The game logic would also be implemented on the game server, which we would check if the move is valid and update the game state accordingly. Upon the creation of the game and both players have connected with the `Game Server`, the server would write a record into the `Game DB`. The schema of the table is shown below:

![Game DB](/assets/online-chess-game-db.png){: width="500" height="300" }

The `game_table` records all critical metadata about the game, such as the `player_id`, `status` and `winner` information. Once the record is created, the `status` of the game would be set to `RUNNING` and once one player wins, the `Game Server` would update the `status` to `DONE` and record the `winner` information. You could also notice that we have created a `game_history_table` to record all the moves of the game. This table is optional in this stateful design (but would be critical for the next design choice), it captures a trajectory of the game state (each move that is accepted by the game server) and we could rebuild the game state from this table in certain failure scenarios (we would explain more in the next section).

We could use SQL or NoSQL database to store `game_table` and `game_history_table`. The read/write ratio of the `game_table` is relative low and it is keeping critical information about the game, we could use relational database to handle it (although we loss the flexibility of things such as schemaless, horizontal scalability). For the `game_history_table`, it would expect relative high write QPS as each move needs to be recorded, we could consider using columnar database to handle it (relational database in append only mode would also work).

Having talked about the game state and the storage, let's move our focus onto how players are connected and how the game is played. There are multiple approaches for connection, such as long polling, server sent event and websocket (you could refer [my pervious post]({{ site.baseurl }}{% post_url 2024-04-06-How-to-design-auction-system %}#websocket-and-sse) for a comparison of websocket and SSE). Since all players need to continuously make moves and receive updates (*bi-directional*) during the game, **websocket** would be a good fit in this case (this is different from the auction system's choice due to the scenarios are different, check [here]({{ site.baseurl }}{% post_url 2024-04-06-How-to-design-auction-system %}#auction-bid-place-and-update) for more details).

The `Game Server` would maintain a websocket connection with players. When one player make a move, the client would send a message to the server via the websocket connection. The `Game Server` would then check if the request is valid or not and apply it to its in memory game state. If we would like to record the step into `game_history_table`, then we need to make sure the write is confirmed by the database and then apply the change to in memory game state to keep it consistent. Once the game state is updated, then the server would send the updated game state to both players to make sure both players are in sync.

The benefits of this stateful game server design is that, for each game we have a single brain to decide what the game should be like in the case of concurrent move requests from players. If player A tries to revert move while player B tries to make a new move and they both send out the request at the same time, then the `Game Server` would be the single place to make a decision on which request to accept and which to reject (maybe based on which request first hit the `Game Server`). For example, if player A's revert request hit first, then player B's move request would be rejected as the move would be illegal; the new game state would be sent back to both player and it would still be player A's turn to make move, vice verse. However, any coin has two sides, the beneficial of handling concurrent request gracefully comes with the cost of a single point of failure. We would discuss more on this in the [high availability](#high-availability) section.

Besides the `Game Server`, we also included a `Coordination Service` component. This component is responsible for keeping the liveness information of the `Game Server` as well as playing a role on **service discovery** by storing associated metadata about the `game_id` and `server_id`. The liveness is used for [high availability](#high-availability) discussed later. The **service discovery** is used for the player to rejoin the game after they have been disconnected. Upon the creation of the game, the `Game Server` would also register this serving information into `Coordination Service`, which essentially is a mapping from `game_id` to `server_id`. When player disconnected, they could rejoin the same game as follow:

1. We could cache the server id information in the client side as well (although this is going to expose some internal information about our cluster and expose certain security issue), if player disconnected due to network issue, then they could reconnect to the same `Game Server` by this cached information
2. We could also not caching any server id information in the client side, or the client's application crash and the cached information is all lost. In this case, the player would be directed by **load balancer** to any `Game Server` (which might not be the one that is hosting the game), and the `Game Server` would check if the player's game is on itself. It send a request to `Game DB` to retrieve the `RUNNING` game that the player is still in and then check the `server_id` from the `Coordination Service`. Once obtain the correct `server_id`, it would then redirect the player to connect to that server. And once connected, the game could be resumed from where it is left (the rejoined player would be sent with latest game state to make sure in sync)

On the client side, since the game state is synced from the `Game Server`, we could implement a simple logic that only if the current turn is the player's turn, then we would send player's move to the `Game Server`. This could help us to reduce the volume of the message sent over the websocket connection to save the network bandwidth.

### Pub/Sub Based Architecture

A high level design of this approach is shown below:

![Pub/Sub Based Architecture](/assets/online-chess-pub-sub.png){: width="700" height="400" }

In this design, there is no `Game Server` to maintain the game state. Instead, all player's move are represented as an event and published to a message queue and the game state is obtained by applying this sequence of events from beginning to end. Here we would adopt message queue that support message persistence, such as *log based* message queue, so that we could support certain failure mode such as player disconnection.

Upon the creation of the game, both players would connect to a `Websocket Server`, which is responsible for keeping the websocket connection with the player, and publish/subscribe to the topic that stores each move. Along with the creation of the game, a dedicated topic for this game would also be created in the message queue, with the format such as `chess-game-{game-id}`. The `Websocket Server` would publish player's move to this topic and also subscribe to this topic to obtain other player's moves. When player makes a move, a message would be sent over the websocket connection to the `Websocket Server` and then publish to the topic. To guarantee the consistency of the game, `Websocket Server` would wait for *ack* from the message queue to make sure that the message is published successfully. Once `Websocket Server` received one new message from the topic, it would relay this message to the player via the websocket connection. From the client side, after sending the move to the server, it would wait for certain threshold time to receive the message back from the server. This message could be the player's pervious move, or the opponent's concurrent move. Based on what is received, game state is changed and corresponding logic would be applied (e.g. lock player's move and wait for opponent's move, game is done or still player's turn and pervious move is ignored). Upon timeout, the client would retry sending the move again.

Notice that in this design, there is no game server on our backend that have a *current* view of the game. We delegate this responsibility to the client side, which means that the client would maintain the game state, as well as game logic such as checking if the player's move is legal or not. Once one player win, a special message would be sent to `Websocket Server`, and the server would update the `game_table` in `Game DB` to record the winner information.

One benefits of using message queue to record the event is that, we could still maintain a single view of the sequence of the events happened and avoid the potential *split brain* problem in the world of concurrency (but this also depends on message queue's replication strategy). Let's discuss a little bit about different scenarios and see how we could still maintain a valid game state:

- Player A tries to revert move while Player B tries to make a new move
  - In this case, we would write 2 message into the message queue: one is a revert message and one is a new move message. Even though they are conflicting with each other, the single topic would still guarantee that only one sequence would be stored, `player a, player b` or `player b, player a` (depending on which message first get to the message queue broker); thus on the client side, we would still receive the same sequence, and thus we could have logic to ignore the invalid message to make sure both player's are in sync and the game state is valid
- Player A made a move, but there is some network delay to receive if the message is published successfully, and thus retried; in the meanwhile, Player B received A's move already and made a move
  - This is a slightly more complicated concurrency issue. There are 2 possible approaches to solve it: 1. use the vector lock to capture the casual relationship among the events and ignore the message that is invalid; 2. use an idempotence key and deduplicate the message that has been processed already. Both solution works, one add additional complexity to the game logic and the other add complexity to the design as additional component need to be added.
  - With vector lock, the player would bump the version number from the previous player's version. In the above scenario, we would see message sequence like the code below in the topic. Retried message could be ignored as the version number is smaller than the pervious one

  ```python
    [
        (p1-1, move-1), 
        (p2-2, move-2),
        (p1-3, move-3),
        (p2-4, move-4), # delay happens here
        (p1-3, move-3), # retry
        (p1-3, move-3), # retry
        ...
    ]
  ```

  - With idempotence key, we could use a simple cache to record if the move has been already processed or not and avoid publish the retried message into the topic
- Player A made a move, received the message back and wait for Player B's move, but Player B haven't received Player A's move yet and is waiting for Player A's move
  - In this case, it is similar to a deadlock situation where both player is waiting for the move due to the out-of-sync. The good news is that there is no new message being written into the topic and the game state is still valid. The reason of the deadlock could be that the `Websocket Server` connected with player B is temporarily portioned from the message queue broker. We could add a retry mechanism in the client side if there is no new message received from the server, we terminate the current websocket connection and establish a new one through *Load Balancer*.

In this design, once player is disconnected, they don't have to connect to the original `Websocket Server` as the game state is stored in the message queue. The player could connect to any `Websocket Server`, and query `game_table` to retrieve the ongoing `game_id` information, and then subscribe to the topic, consuming the messages from the beginning of the topic to rebuild the game state.

As message queue is not intended for persistent storage, we could move the message to a database table such as `game_event_table` to free up the resource in message queue. This could be done via a offline daily job, that scan the `game_table` to find all game that has already been finished, and them move the message data from the message queue to `game_event_table`. This could be done via a dedicated in-house service, or some solution offered by the eco-system of the message queue.

## High Availability

In this section, let's discuss how to achieve high availability for our online chess game.

### Game DB

The `Game DB` is one critical component as it stores all metadata about the game. We need to make sure the data stored is replicated to avoid it being the single point of failure. There is different strategy to achieve this: **Single Leader Replication**, **Multi Leader Replication** and **Quorum Based Replication**, as well as **Synchronized** and **Asynchronized** message replication among the replicas. I would not expand too much on this topic, you could refer to my video on [DDIA Chapter 5](https://www.youtube.com/watch?v=X8IhZx7fg24) for more details.

{% include embed/youtube.html id='X8IhZx7fg24' %}

### Message Queue

The message queue is another critical component in the **pub/sub based architecture** as all events data are stored there. We could use the similar strategy we used in `Game DB` to achieve high availability here. However, if we adopt **Multi Leader Replication** or **Quorum Based Replication**, there could be write conflict (e.g. player A's revert and player B's move) and it is challenging to resolve it (we could not directly merge them like a shop cart, nor we could ask client to do a custom fix as there are 2 players). Using **Single Leader Replication** would be a good choice for us to easier handle the consistency of the game, which is critical to online chess game. Also if we use **Asynchronized** replication, we might loss some events as the message is not replicated from leader to followers yet. Players would see that the game state transition back to the pervious status, and this might cause some issue especially in such turn-based competing game (you read the opponent's move :joy_cat:). So in this case, adopt **Synchronized** replication might be a better option.  

### Game Server

The `Game Server` component is critical in the **stateful game server** design as it is maintaining the state of the game. There are several options to make it fault tolerant:

- **WAL**, this might be useful in the case that the `Game Server` is managing each game via multiple threads or processes, and the thread/process crash accidentally. The `Game Server` in this case could quickly rebuild the game state by reading the content in **WAL** file
- **KV Cache**, if the `Game Server` crash, then we need to store the game state somewhere aside from the server to make it possible to resume the game from a new server instead of waiting for the original server to become alive. We could use a **KV store** to store the game state periodically. If the original game server crash and we need to have a new server to take over, the new server could read the game state from the **KV store** and resume the game
- **Replicate on other Game Server**, nothing blocks us from replicating the game state to other game servers similar to what we do for `Game DB` and message queue. We could run a simple *leader election* upon the creation of the game to decided which server would be the leader to handle all the read/write requests, and replicate the game state some some other selected game servers. These information would be stored in `Coordination Service`. Also each game server would also send heartbeat to `Coordination Service` for the liveness check. If one replica find that the leader is dead, it could initiate a new leader election and nominate itself as the new leader. Once new leader is elected, players (right now should be retrying connecting) would be redirected to the new leader and the game would resume from there. Similar to the discussion in the above section, adopting **Synchronized** replication might be a better option to avoid the traverse back issue. If we don't use the `Coordination Service` to handle the leader election, we could also run a consensus algorithm such as RAFT under the hood to select a server as the leader of a game and other servers could be followers. However, running consensus algorithm on a large cluster would cause high performance cost.

Adopting **KV store** seems to be a attractive option, but there are some additional points we should bear in mind to have a more comprehensive understanding on this option. One thing is the frequency to store the game state, a.k.a snapshotting. We could snapshot at each move, but this would cause large delay; or we could snapshot after certain batch of moves or certain period of time, but this has the risk of losing part of the game state. This is a trade off we need to discuss based on the scenario. In the case of online chess game, since majority of time is spent by players thinking about their next move, thus the delay incurred by snapshotting after each move is relatively low.

In both the **KV store** and **Replication on other Game Server** approach, it is critical to have a single server as the leader on the game and have the single authority to determine the game state as well as store it. This is the reason why we have `Coordination Service` to help achieve this. **Replication on other Game Server** could leverage the `Coordination Service` for the leader election (e.g. [use sequential/ephemeral node in Zookeeper](https://zookeeper.apache.org/doc/current/recipes.html) or use [etcd leader election API](https://github.com/etcd-io/etcd/blob/main/client/v3/concurrency/election.go)). In the **KV store** approach, the `Coordination Service` would act as a lease manager to make sure that at any time, only one server could hold the lease and thus be the leader of the game. For example, if `Game Server A` is the current leader part suddenly portioned from the `Coordination Service`, the `Coordination Service` would wait for certain time to make sure the lease has passed to guarantee that any remaining write from `Game Server A` has been applied to the `KV cache`, and then a new server could acquire the lease and become the new leader, start reading the snapshot from the `KV cache` and resume the game. We also need to be able to pause the game from the old server and redirect the player to the new server, this could be done by having a check to see if the game server still receive heartbeat ack from the `Coordination Service`, if not, then after the lease timeout, it would disable the connection it is holding so that players would trigger the reconnection logic to be direct to the new game server.

### Coordination Service

This is usually some third party service that is running on a small quorum of nodes with some distributed consensus algorithm such as [RAFT](https://raft.github.io) to achieve high availability. Discussion into the details of these algorithm is usually out the scope of traditional system design interview. But I still recommend to learn about these algorithm as they are critical to understand the underlying principle of distributed system.

Here is a final view of the **stateful game server** design with high availability consideration:

![Stateful Game Server with High Availability](/assets/online-chess-game-server-ha.png){: width="700" height="400" }

### Websocket Server

There is no special treatment for the `Websocket Server` to make it highly available. As their primary goal is to keep the websocket connection with players. If one server is down, the player could connect to another server and resume the game from there.

## High Scalability

The scalability of the 2 design choice is slightly different. The **pub/sub based architecture** is more scalable due to the following factors:

- The decoupling of player connection and game state storage. As the number of players increasing, the number of websocket connections and the game state need to be stored also increase. In **pub/sub based architecture**, as we have decoupled these 2 properties into different components, we have the freedom to scale them independently. While in the **stateful game server** design, these 2 properties are coupled on the same game server and make it hard to scale appropriately

In both design, the data could be distributed across multiple nodes by sharding on `game_id` with hashing functions. Key range based sharding might not be a good choice here due to the imbalanced distribution of the data. For all the tables we stored in `Game DB`, such as `game_table` and `game_event_table`, we could also shard them by the `game_id`.

## Comparison

Overall, **stateful game server** is relative easier to maintain and achieve the consistency requirement; while **pub/sub based architecture** is more scalable.

|      | Stateful Game Server | Pub/Sub Based Architecture |
|------|-----------|-----|
| Game state | Final state| Sequence of events |
| Complexity | Low, easier to reason about | High, need to reason about the events and handle the potential illegal ones |
| Latency | Low, single round of connection | Higher, 2 step of connection |
| High Availability | Achieved by snapshotting game state | Achieved by replication strategy of message queue |
| Scalability | Need to trade off between connection, CPU/Mem usage | Easier to scale horizontally and individually |
| Debugability| Low, need additional setup such as `game_history_table` | High, natively support that |

> If you find this post helpful, feel free to scan the QR code below to support me and treat me to a cup of coffee
{: .prompt-tip }

![Thank You](/assets/qr%20code.png){: width="300" height="300" }

## Reference

- [Zookeeper](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- [Etcd](https://etcd.io)
- [我用消息队列做了一个联机小游戏](https://mp.weixin.qq.com/s/kI0HUTFVr4YEBpLRZWLEDg)
- [Raft](https://raft.github.io)
- [Python Websocket](https://websockets.readthedocs.io/en/stable/intro/tutorial1.html)
- [Pulsar](https://pulsar.apache.org/docs/3.3.x/concepts-architecture-overview/)
