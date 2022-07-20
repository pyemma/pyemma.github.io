---
layout: single
title: MIT Distributed System Course - KVRaft
tags:
- distributed system
---
It has been a long time since last update on the project. Finally I have found some time that I could resume this project and finish it up.

In this post, I would mainly introduce the work on the LAB 3A, which is to leverage RAFT we have implemented in LAB 2 to build a reliable distributed key-value store. Before jumping into this part, I would also highlight some change to my RAFT implementation. I haven't used GO language in my daily work a lot, and still adopting lots of philosophy in Python, which makes my implementation not elegant. I have learnt from ideas from online resources, which not only makes the code more readable, but also more reliable to pass the tests in the project.

## Update on RAFT

In the pervious [post](https://pyemma.github.io/Distributed-System-RAFT/), I have used lots of different *background go routines* to repeatedly checking if certain condition is meet and we need to trigger some actions. For example:

* A go routine to check if the leader election timeout needs to be triggered or not. This logic is jointly coupled with the AppendEntries/RequestVote RFC API where we need to handle the reset of the timer
* A go routine to periodically send replica log request or heartbeat signal to other nodes, if the current node is the leader
* When sending RequestVote rpc call, we start a new go routine, and use a condition variable to check if we have collect enough vote

This implementation is not that elegant, as it losses a "causal relationship" among different events. After searching a little bit on the web, I learnt a new approach to implement, which is to use *channel* as the media to pass signals, and use *select* to organize the events that is happening concurrently.

Here is a code snippet of the new design
{% highlight go %}
func (rf *Raft) startBackgroundEvent() {
  for {
    rf.mu.Lock()
    state := rf.state
    rf.mu.Unlock()

    switch state {
    case Leader:
      select {
      case <-rf.backToFollower:
      case <-time.After(HeartbeatTimeout):
        rf.mu.Lock()
        rf.broadcastAppendEntries()
        rf.mu.Unlock()
      }
    case Follower:
      select {
      case <-rf.votedCh:
      case <-rf.heartBeatCh:
      case <-time.After(rf.getElectionTimeout() * time.Millisecond):
        rf.convertToCandidate(state)
      }
    case Candidate:
      select {
      case <-rf.backToFollower:
      case <-rf.winCh:
        rf.convertToLeader()
      case <-time.After(rf.getElectionTimeout() * time.Millisecond):
        rf.convertToCandidate(state)
      }
    }
  }
}
{% endhighlight %}

For example, when follower make a vote upon receiving RequestVote RPC call from candidate, if we vote, then we would send a signal over the votedCh, this would suppress the election timeout, similar case when we receive heartbeat on AppendEntires RPC. For leader, unless it receive signal over backToFollower channel, which would be send if certain condition is met during handel RPC call response, it will periodically send the AppendEntries call to all nodes.

In this change, I also move the leader commit and apply command on channel from background routine to be part of functions to broadcast AppendEntries RPC call.

{% highlight go %}
func (rf *Raft) sendAppendEntriesV2(server int, args *AppendEntriesArgs, reply *AppendEntriesReply) {
  ok := rf.peers[server].Call("Raft.AppendEntries", args, reply)

  if !ok {
    return
  }

  rf.mu.Lock()
  defer rf.mu.Unlock()

  if reply.Term != rf.currentTerm || rf.state != Leader {
    return
  }

  if reply.Term > rf.currentTerm { // at this time we need to step down
    rf.convertToFollower(reply.Term)
    return
  }

  if reply.Success {
    matchIndexNew := args.PrevLogIndex + len(args.Entries)
    if matchIndexNew > rf.matchIndex[server] {
      rf.matchIndex[server] = matchIndexNew
    }
    rf.nextIndex[server] = rf.matchIndex[server] + 1
  } else {
    rf.nextIndex[server] = rf.updateNextIdx(reply)
    rf.matchIndex[server] = rf.nextIndex[server] - 1
  }

  rf.updateCommit()
  go rf.applyLogs()
  return
}
{% endhighlight %}

With the new design of the code, the test in lab2 could be passed more reliably.

## Build Distributed Key-value Store over RAFT

For the next part, let's go over some details on how to build a distributed key-value store over RAFT. All of the code could be found in this [repo](https://github.com/pyemma/mit-distributed-system/tree/master/src/kvraft).

Overall, the architecture of the key-value store is that

* Each kv server has a raft node peer, the kv server would only talk to its raft peer and no other communication. 
* Client would make RPC call to kv server. It would only talk to the server whose raft peer is leader, and it may take sometime for client to figure out who is leader. In this lab we just use round robin's approach to check who is leader, in production we might consider establish such info into zookeeper for client to quickly know who is leader and if there is leader change
* Upon each request received, kv server leader would submit the command to raft node for log replication. And it is going to listen on the applyCh channel to see if the command from some client's request has been handled or not. Listening on applyCh for command to execute and response to the RPC call from client is happening on different go routines (one we created, one created based on how go handle RPC call). To coordinate them, we use channel to send signal.
  * We use the command index returned from raft Start() function as the identifier for our request and register a channel on it (using a map structure). In the listening routine, we read the command, execute it, and send the signal to the channel retrieved from the map.
 {% highlight go %}
  kv.mu.Lock()
  ch, ok := kv.channels[idx]
  if !ok {
    ch = make(chan Result, 1)
    kv.channels[idx] = ch
  }
  kv.mu.Unlock()

  select {
  case <-time.After(time.Millisecond * 6000):
    reply.Err = ErrWrongLeader
  case res := <-ch:
    if res.ClientId != op.ClientId || res.RequestId != op.RequestId {
      reply.Err = ErrWrongLeader
      return
    }

    reply.Err = OK
    kv.mu.Lock()
    delete(kv.channels, idx)
    kv.mu.Unlock()
  }
 {% endhighlight %}

{% highlight go %}
 // start the background thread to listent to the applyCh
 // to get the committed op and mutate the kv store
 go func() {
    for msg := range kv.applyCh {
      op := msg.Command.(Op)
      idx := msg.CommandIndex
      res := Result{
        Err:       "",
        Value:     "",
        CmdIdx:    msg.CommandIndex,
        ClientId:  op.ClientId,
        RequestId: op.RequestId,
      }
      // start to handle the committed op

      // handle the duplicated request, by checking the request id
      lastId := kv.lastRequest[op.ClientId]
      if op.RequestId > lastId {
        kv.lastRequest[op.ClientId] = op.RequestId
        if op.Type == "Get" {
          val, ok := kv.store[op.Key]
          if ok {
            res.Value = val
            res.Err = OK
          } else {
            res.Err = ErrNoKey
          }
        } else if op.Type == "Put" {
          kv.store[op.Key] = op.Value
          res.Err = OK
        } else {
          val, ok := kv.store[op.Key]
          if ok {
            kv.store[op.Key] = val + op.Value
          } else {
            kv.store[op.Key] = op.Value
          }
          res.Err = OK
        }
      } else {
        res.Err = OK
        if op.Type == "Get" {
          val, ok := kv.store[op.Key]
          if ok {
            res.Value = val
          } else {
            res.Err = ErrNoKey
          }
        }
      }

      kv.mu.Lock()
      ch, ok := kv.channels[idx]
      if !ok {
        ch = make(chan Result, 1)
        kv.channels[idx] = ch
      }
      kv.mu.Unlock()
      DPrintf("Finish processing one result %s, %s, %s, client %d, request %d, server %d", op.Type, op.Key, op.Value, op.ClientId, op.RequestId, kv.me)

      ch <- res
    }
  }()
{% endhighlight %}

  * There is a timeout for each cleint's RPC call. Client would resend the request to other server if there is timeout. However, we should only execute the "Put" and "Append" request only once. Sometime server might have already commit the command, but timeout and fail to response to client. This request us to have a mechanism of duplication identification. The solution I adopted is to attach a request id to each client's request, and on the server we hold the latest request id we have executed. We directly skip the duplicated "Put"/"Append" request by checking the client id and request id.
* For each follower, their raft peer receive the replicate their log according to leader, and return command committed to the applyCh. KV server just execute these commands and there is no need to handle the client request.

Initially my implementation could not pass all the test cases in the lab reliably. After some search on the web, I found the following 2 is the most critical part of the implementation

### Use command idx as the key for channel to signal RPC handler

In the beginning, I was using client id + request id as the identifier of the channel. However, this approach is hard to manage correctly. Command idx is universally unique among RAFT, and use it as the identifier would greatly simply the management logic to signal RPC call.

Also, one nits that simply the logic to manage channel a lot is that, we create the channel in the routine that is listening on applyCh channel and apply command to kv server's state. Although the channel might be outdated and no RPC call is waiting on that, it could help avoid a accidentally sending signal on closed channel.

### Adding timeout in the RPC handler

Although in the lab statement there is no explicit ask to add timeout on the RPC call, I found it is one of the most critical mechanism to implement to make the kv server working.

One corner cases I have found could not be mitigated without timeout is as follow:

* Sometime the network is partitioned, and once the pervious partitioned leader comes back, there might be a new round of leader election. In this case, we could hit a situation that, the current leader still becomes the leader, but we are in a new term right now
* And then, if during this change, there is a RPC call happens and is waiting for response, without timeout mechanism, this request might waiting indefinitely. And the entire system might freeze and making no progress.
  * The reason of this issue is that, leader could not commit the entires in previous term. And a corner cases could happens that, although leader has replicate the logs all on followers, but just before leader update the commit index to commit the log, leader selection happens and we are in a new term. And although all follower has exactly same log with leader, leader could not update commit index because leader could only commit ones in its own term.

By adding timeout mechanism, we could break the above issue that. Client would send the same request again, and leader would add it to our log as well. Although on the log, we would have duplicated command, we have already added the dedup mechanism in our system to handle it. And since client send a new request, it would be added as a new log in new term. Once leader confirmed that this command has been replicated on all followers, it could commit all logs before it (including the duplicate one in pervious term), and thus the entire system could make progress smoothly.

Debugging this corner case is really challenge. I would never forget the excitement when I finally found and understood the root cause.
