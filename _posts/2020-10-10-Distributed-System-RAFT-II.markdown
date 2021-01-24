---
layout: single
title: MIT Distributed System Course - Raft II
tags:
- distributed system
---

## Introduction

In this post, we continue our work on raft implementation. The focus of this post would be the second part of raft, which is the log replication, and snapshot to optimize the size of the log.

Let's first go through the high level logic of how log replica works in raft:

* When the leader sends out a log replica request, it is going to also send the new log entries' pervious entry's term and index along the request, as well as the current committed entry index
* Upon receiving the request, followers/candidates would perform the regular check as usual
* Then, followers would check the pervious entry's term and index send by the leader to see if there is any conflict with its own log
* If there is no conflict, then followers would copy over the new entries sent by the leader, otherwise it would reject this request
* Once the leader receive the reply from a follower, if it success, then it would update its nextIndex (the next index need to be send to a particular server) and matchIndex (the highest index to be known have been replicated on a particular server); if it fails, then the leader would reduce the nextIndex and retry the request
* Periodically, leader would check the matchIndex and see if there could be new log to be committed
* If a follower see that the commit index passed from leader is greater than its current commit index, it would update its current commit index as well

From this entire process, leader would never change its own log entries and acts as a dictator to ask followers to replica its authority. Also, leader would never commit log entries from past terms to show respect to former dictators :). It always commits log from its current term, but the entries from past term would be implicitly committed.

The section 5.4.3 has provided a pretty good explanation on the safety guarantee of the raft: *Leader Completeness Property*. Below is the full list of RAFT properties introduced in the paper.

![RAFT properties](/assets/raft_property.png)

## Log Replica

Two functions plays a key role on Log Replica: updated version of `AppendEntries` by followers and `replicaLog` by leaders.

We have already provided an implementation of `AppendEntries` in the last post so that the leader could use it to send heartbeats to followers. We extend the functionality of `AppendEntries` to make it support log replica.

Here is the new implementation of the function (or take a look at [highlight of the change](https://github.com/pyemma/mit-distributed-system/commit/4ca09b83b23325f14bfceae8161f366a4ddc030d)).

{% highlight go %}
func (rf *Raft) AppendEntries(args *AppendEntriesArgs, reply *AppendEntriesReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	if args.Term < rf.currentTerm {
		reply.Term = rf.currentTerm
		reply.Success = false
		return
	}

	if args.Term > rf.currentTerm {
		rf.convertToFollower(args.Term)
	}

	if args.Term == rf.currentTerm && rf.state == Candidate {
		rf.convertToFollower(args.Term)
	}

	rf.resetElectionTimer()

	reply.Success = true // default to true, all the logic below would set it to false
	reply.Term = rf.currentTerm

	DPrintf("Server %d, args perv log %d, args perv term %d, my log %v", rf.me, args.PrevLogIndex, args.PrevLogTerm, rf.logEntries)
	pervCheck := rf.checkPerv(args.PrevLogIndex, args.PrevLogTerm)

	if pervCheck == false {
		reply.Term = rf.currentTerm
		reply.Success = false
		rf.updateXInfo(args.PrevLogIndex, args.PrevLogTerm, reply)
		DPrintf("XTerm %d, XIndex %d, XLen %d", reply.XTerm, reply.XIndex, reply.XLen)
		return
	}

	if len(args.Entries) == 0 {
		rf.checkCommit(args.LeaderCommit)
		return
	}

	rf.checkAndCopy(args.PrevLogIndex, args.Entries)
	rf.checkCommit(args.LeaderCommit)
	return
}
{% endhighlight %}

The key new logic we introduced is as follow:

* We introduce a new function, which is `checkPerv`. This function is used to check if there is any conflict entry from follower's current log with the pervious log entry from leader. Note that even if the request is a heartbeat, where there is no new entry sent from leader, we still needs to do this check as described in the algorithm, so that follower could pass some useful information back to the leader to update the leader's internal state about followers
* We also introduce a function called `checkCommit` to see if the leader has send a commit index that is larger than follower's current commit index on record
* Once the pervious entry check is pass, we use the function `checkAndCopy` to do the actual copy and overwriting work, where we find the first entry that is different with the new entries sent by leader, and copy-paste from that point.

As we have seen the `AppendEntries` RPC call's logic, let's take a look at how leader is going to send request to followers and process responses from followers. In the pervious post, we have implemented a function called `sendHeartbeat`. We rename it to `replicaLog` and extend its functionality, with a parameters to control whether we would love to use it to send heartbeats, or send real log entries. Below is the implementation (or take a look at [highlight of the change](https://github.com/pyemma/mit-distributed-system/commit/4ca09b83b23325f14bfceae8161f366a4ddc030d)).

{% highlight go %}
// replicaLog is the function used by leader to send log to replica
func (rf *Raft) replicaLog(isHeartbeat bool) {
	term := rf.currentTerm
	if isHeartbeat == true {
		rf.heartbeatTime = time.Now() // rest the heartbeatTime
	}
	rf.mu.Unlock()

	for peer := range rf.peers {
		if peer == rf.me {
			continue
		}

		go func(server int, term int, isHeartbeat bool) {
			rf.mu.Lock()
			if rf.state != Leader || rf.currentTerm != term {
				rf.mu.Unlock()
				return
			}

			nextIdx := rf.nextIndex[server]
			lastIdx := len(rf.logEntries) - 1
			if lastIdx < nextIdx && isHeartbeat == false { // in this case, we have nothing to update
				rf.mu.Unlock()
				return
			}

			perLogIdx := nextIdx - 1
			perLogTerm := rf.logEntries[perLogIdx].Term

			entries := rf.logEntries[nextIdx:]
			if isHeartbeat {
				entries = make([]LogEntry, 0)
			}

			args := &AppendEntriesArgs{
				Term:         term,
				LeaderId:     rf.me,
				PrevLogIndex: perLogIdx,
				PrevLogTerm:  perLogTerm,
				Entries:      entries,
				LeaderCommit: rf.commitIndex,
			}

			reply := &AppendEntriesReply{}
			rf.mu.Unlock()

			ok := rf.sendAppendEntries(server, args, reply)

			if ok {
				rf.mu.Lock()
				DPrintf("%d append entries get reply from %d, %t on term %d, is heartbeat? %t", rf.me, server, reply.Success, reply.Term, isHeartbeat)
				if reply.Term > rf.currentTerm { // at this time we need to step down
					rf.convertToFollower(reply.Term)
					rf.mu.Unlock()
					return
				}

				// check if the condition still matches when we schedule the RPC
				if reply.Term == rf.currentTerm && rf.state == Leader {
					if reply.Success == true {
						rf.matchIndex[server] = lastIdx
						rf.nextIndex[server] = lastIdx + 1
					} else {
						// need to do an optimization here
						// rf.nextIndex[server] = rf.nextIndex[server] - 1
						rf.nextIndex[server] = rf.updateNextIdx(reply)
					}
				}

				rf.mu.Unlock()
				return
			}
		}(peer, term, isHeartbeat)
	}
}
{% endhighlight %}

The key change is:

* We read the next index of the entry we need to send to follower *X* from nextIndex, which is initiated inside the leader when it is voted. This records the next index of the entry we need to send to each follower. Leader initiated it to the last index + 1. All entries after the next index is the new entries we would love follower *X* to replica
* We also retrieve the pervious entry of the next index entry, send its index and term along the request
* Upon receiving reply, we update the matchIndex and nextIndex accordingly. If request is rejected, we reduce the nextIndex of the follower *X* by one, this is a linear search of the first entry in leader's log that the follower *X* would agree. However, in this assignment, it requires us to do some smarter search instead of linear search to speed up.

The final piece of the logic is to create a thread to send replica to other servers. This is pretty similar to the thread that we periodically send heartbeats.  

## Commit Log and Apply

Another job we need to do is to commit log entries. The logic is done by the leader with the helper function `updateCommit`. Similarly, we also create a dedicated thread to periodically call this helper function to see if we should advance the current commit index. The logic of `updateCommit` is pretty simple, it just iterates all possible commit index and find the highest one that is qualified, which means that there are majority of server's highest match index greater than it.

{% highlight go %}
func (rf *Raft) updateCommit() {
	// DPrintf("leader %d, current match index %v, current commit index %d", rf.me, rf.matchIndex, rf.commitIndex)
	newCommit := len(rf.logEntries) - 1
	for ; newCommit > rf.commitIndex; newCommit -= 1 {
		commitCount := 1
		for _, match := range rf.matchIndex {
			if match >= newCommit && rf.logEntries[newCommit].Term == rf.currentTerm {
				commitCount += 1
			}
		}
		// DPrintf("leader %d, current commit index %d, new commit index %d, commit count %d", rf.me, rf.commitIndex, newCommit, commitCount)
		if commitCount >= (len(rf.matchIndex)/2 + 1) {
			rf.commitIndex = newCommit
			break
		}
	}
}
{% endhighlight %}

Once the current commit index is updated, there is another thread working in background that going to apply the log entry from the current applied to the current commit index.

## Snapshot

In the final part of the Lab2, we are asked to add the functionality to persist state, which is the well known practice snapshot. According to the algorithm, there are in total three values need to be persisted: currentTerm, votedFor and log[]. Although we usually persist state to disk, in this lab we mimic this by using a dedicated class to encode and decode state. The change could be viewed [here](https://shorturl.at/A1278). Anytime we make a change to the above 3 state, we would call the persist function to create a snapshot. When a raft server crashed, it would read the snapshot it has persisted and start to catch up others from there, instead of starting everything from the beginning, which could be pretty time consuming.

## Future work

By this point, we have implemented the Raft algorithm E2E. Working on Raft has been one of the most challenging coding assignments I have ever had, especially debugging the abnormal behavior of the system. In the following lab, we are going to build other distributed system on top of the raft algorithm we have implemented here. Stay tuned!