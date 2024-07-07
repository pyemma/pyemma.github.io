---
layout: single
title: 读书笔记 - Patterns of Distributed System
tags:
- distributed system
- system design
- 读书笔记
toc: true
---

最近读了一本和 distributed system 相关的书籍，介绍了在 distributed system 里面常用的一些 pattern. 这是一篇简要的读书笔记，把书中提到的几个 pattern 总结了下来; 我计划会经常更新这篇 blog, 把我新学习到的或者总结出来的一些 pattern 记录在这里; 希望能起到一个引导性的作用，给大家提供一个提纲挈领的思路

## Patterns

### Write Ahead Log

把命令存储到一个 append only file 里面去，当挂了之后可以重新读 WAL 来 rebuild 内部的 state #Message-Queue #KV-Store #持久化

- Flushing 来保证命令真的写到 physical media，好处是 persistent，代价就是 performance; 可以使用 batching 等方法来进行优化 #Batching
  - CRC record 来防止 corrupted entry #CRC
  - Log 里面可能有 duplication，每一个 request 需要一个 unique identifier 来进行区分 #Deduplication
  - 可以用来实现 transaction，用来保证原子性 #Transaction
  - 工业界里面的具体例子 #RocksDB #Kafka #Cassandra
  - Key/Value pairs that needs atomic store, write into a batch, and then batch is add into data store; the data store first create a WAL entry for the entire batch, once log is created successfully, the batch is added into datastore

### Segmented Log

把单一的 log file 切分成更多的 log 从而方便对老的数据进行 cleanup; 当数据超过一定的阈值之后就 rollover 到一个新的 log file 里面去, 业界的例子 #Kafka #Cassandra #Raft

### Low Water Mark

帮助保证 log 的大小不会无限制的增长，通过 low water mark 这样的一个 index，对 log 进行压缩 (通常是一个 background job 在进行这个操作)

- Snapshot-based #Raft
- Time-based #Kafka

### Leader and Follower

使用单一的 server 来 coordinate 多个 servers 的 replication #Replication

- Small cluster: leader election, #Zab #Raft
- Large cluster: consistent core, 需要的几个核心功能 #Zookeeper #etcd
  - compareAndSwap to set a key atomically
  - heartBeat to expire the key if no heartBeat from leader, and trigger new election
  - notification mechanism to notify all servers if key is expired

### Heartbeat

- 可以使用 separated thread 来异步发送 heartbeats  #Consul
- 在 large cluster 里面，1-to-1 的 heartbeat messaging 效率太低了，这个时候一般可以考虑使用 Gossip Protocol #Gossip-Protocol
  - 两种主流的实现方式，Phi Accrual failure detector 和 SWIM #Cassandra #Consul

### Majoruty Quorum

Flexible quorum, 我们可以通过动态的调整读写的 quorum size 来提高性能，只要能保证读写之间会有一个交集就行; 比如说一共有 5 个 node，然后我们有 90% 的读和 10% 的写，那么我们可以要求读只需要 2 个 quorum, 写需要 4 个 quorum #Quorum #Cassandra

### Generation Clock

也可以叫做 Term, Epoch, 这个是 Lamport Clock 的一个具体样例 #Lamport-Clock

- Each process maintains an integer counter, which is incremented after every action the process performs. Each process also sends this integer to other processes along with the messages processes exchange. The process receiving the message sets its integer counter by choosing the maximum between its own counter and the integer value of the message. This way, any process can figure out which action happened before the other by comparing the associated integers. The comparison is possible for actions across multiple processes as well, if the messages were exchanged between the processes. Actions which can be compared this way are said to be *causally related*.
  - 工业界的例子, Cassandra 里面的 server 在 restart 的时候会自增 1, 这样在 gossip 的 message 里面其他的 server 会知道这个 server restart 了，从而会把关于这个 server 的 stale 的 data drop 掉，然后要新的; Kafka 里面的 epoch number 会存在 Zookeeper 里面，每次一个新的 controller 被 elect 的时候，就会增加这个 epoch number; 同时 leader 也会 maintain 一个 Leader Epoch 来看是否有 follower 太落后了 #Cassandra #Kafka

### High Water Mark

也被称作是 **CommitIndex** #Replication #Raft #Kafka

- Client 最多只能读到这里，因为在 high water mark 之后的 entry 都还没有被 confirm 已经 replicate 了
- 这个在 stream 里面处理 delayed event 时候也叫这个，只不过那个 high water mark 是多等一段时间

### Paxos

这个太难了，等以后专门开一个总结一下吧 #Paxos #Consensus-Algorithm #2PC #Quorum #Lamport-Clock

- We can ensure liveness or safety, but not both. Paxos ensure safety first
- 工业界的具体应用: Google Spanner 使用的是 multi-paxos, which is implemented as a replicated log; Cassandra uses basic Paxos to implement lightweight transactions #Spanner #Cassandra

### Replication Log

- 在 MongoDB 中，每一个 partition 会有一个自己的 replication log #MongoDB #Partition
- 在 Kafka 的 Raft 实现中，使用的是 pull 模式，也就是 follower 从 leader 那里 pull replication log #Kafka #Push-Pull
- Read request optimization via bypassing the replication log, 可以使用两种不同的方法, 一个是 leader 再发送一个 heartbeat 然后看能不能得到 majority 的回复，来确认自己仍然是 leader; 另一个是使用 leader lease #Read-Optimization #Lease #etcd

### Idempotent Receiver

client 可能会 retry request, server 端需要进行 deduplication, 这个在多种系统中都很常见 #Event-Aggregation #Payment

- 给每个 client 一个 unique id, 在 server 端进行注册，注册之后 client 才能开始给 server 发送 request; 这个数据也需要被 replicated 从而保证高可用性
- Expiration of saved request, request number, next request only when received response, number of max in-flight request with request pipeline #Kafka

### Singular Update Queue

一种用来高效处理 concurrent request 的方法，向比较使用 lock 的话效率更高；具体的实现方法就是实现一个 work queue, concurrent 的 request 都放到 queue 里面，但是只有一个 worker thread 来处理 queue，从而实现 one-at-a-time 的保证 #Concurrency #Message-Queue #Coordination

- 工业界的例子有 Zookeeper, etcd, Cassandra
- 可能会用到这个思想的 system design: Booking, Google doc (OT)

### Request Waiting List

一个 node 可能要和其他的 node 进行异步的 communication 之后才能返回 request, 保存一个 waiting list 来 map 一个 key 和一个 callback function #Concurrency #异步

- 工业界例子: Kafka 里面的 purgatory 来保存 pending request #Kafka

### Follower Reads

也就是大名鼎鼎的 read replica; 即使是在用 Raft 这种 consensus 算法来进行 replication 的系统中也会有 replication lag, 因为 leader 需要一个 additional 的 network call 来让所有的 follower 都 commit; read your own write，可以使用 lampart lock 来解决，写了之后传回去一个 version number，再读的时候要带着这个 version number 来看 read replica 上面的 value 是不是已经是更新的了

### Version Number

To store versioned key values, a data structure that allows quick navigation to the nearest matching version is used, such as a skip list, 之前在 lucene 里面也看到了这个 skip list，需要研究一下 #数据结构

在 RocksDB 里面，一个重要的原因需要把 key sorted 的是因为它们 underlaying 存储的都是 bytes array, its important to keep keys sorted when they are serialized into byte arrays

### Version Vector

在 Cassandra 里面，除了 value 以外，还把 timestamp 也当做一个 column 来存储了，从而实现了 LWW，但是代价就是 Cassandra 的 cluster 需要正确的设置 NTP, 否则的话 latest value 仍然可能被 old value 给 overwrite 掉

- 如果每一个 cluster client 有一个 unique id 的话，那么我们也可以使用 client id 来存 version vector (但是这样的话怎么进行 conflict resolve 呢)
- 一篇 Riak 里面讲针对使用 client id 还是使用 server id 来存储 version vector 的[文章](https://riak.com/posts/technical/vector-clocks-revisited/index.html?p=9545.html)

### Fixed Partition

先 create logic shard，然后再把 logic shard map 到 physical shard 上面去; 这些 metadata 都可以通过一个 coordination service 来负责 (分 partition 和 存储相应的 metadata); 另外一种做法是每个 physical node 上面的 partition 数量是固定的，也就是 propositional to number of nodes
      
- Kafka 里面的每一个  topic 就是一个 fixed size partitions

### Clock Bound Wait
W
hile reading or writing, cluster node wait until the clock values on every node in the cluster are guaranteed to be above the timestamp assigned to the value

- Google TrueTime, AWS Time Sync Service, 使用 atomic clock 和 GPS 来确保 clock drift across their cluster node is kept below a few milliseconds
- 这个概念有点复杂，需要再找一个好的资料学习理解一下这里的思想

### Lease

Use time-bound lease for cluster nodes to coordinate their activities, 这个在 GFS 里面就使用到了, 在 Facebook 的 Memcache 里面也有涉及到 Lease 的思想, Lease 一般可以通过一个 coordination core 来实现，由 leader 来进行 lease 的 replication 和 check

### State Watch

可以参考一下是怎么实现的, 在 server 端我们需要存储下来 event 和 client 的 connection, 在 client 端我们要存储 event 和对应的 handler

### Emergent Leader

直接用在整个 cluster 里面最老的那个 node 作为 coordinator node, 相比较 consistency core 所采用的 leader election 的方法， favor availability over consistency

### Single Socket Channel

在 follower 和 leader 之间保持一个能够支持 retry 而且能够保证 message order 的通讯，可以通过 TCP 来实现; 在 Kafka 和 Zookeeper 里面使用了这种方式 #Kafka

### Request Batch

把多个 request 放在一起从而提高带宽的利用率; 在 client 端可以 maintain 一个 queue 来维护 request, 然后再放在一个 batch 里面一起发过去 (这个其实跟之前写的 batch commit logger 是一样的)

### Request Pipeline

server 在发出去 request 之后不需要等待 response, 又另外一个 thread 来负责接受和处理 response (有点像是 webhook 的思路); 为了防止 request overwhelming, 一般会有一个 upper bound on max in-flight request; 同时针对 retry 和 out-of-order 的 request 也需要针对性的处理 (比如 assign unique request id 等)

## Reference

- [Patterns of distributed system](https://martinfowler.com/articles/patterns-of-distributed-systems/)
