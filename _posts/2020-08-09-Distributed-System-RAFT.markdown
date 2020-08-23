---
layout: single
title: MIT Distributed System Course - Raft I
tags:
- distributed system
---

## Introduction

In the following serious of post, we are going to implement Raft consensus algorithm, which is used to manage replicated log. And on top of that, we would implement a failure tolerance key-value store. This type of failure tolerant system is called replicated state machine. Replicated state machine operates on a collection of servers and makes them acting like a single server to the outside. The service would still be alive as long as majority of servers are still working.

## Raft

The original paper is available [here](https://raft.github.io/raft.pdf) and there is an official [website](https://raft.github.io/). In this post, we would first introduce some basic concepts/abstractions in Raft. Then we would move to the Leader Election part, which is cover by the 5.1, 5.2 and 5.4 section in the original paper.

The below figure in the original paper is a decent summarization of the Raft algorithm, which is recommend to read carefully words by words.

![Summary of Raft consensus algorithm](/assets/raft.png)

There are several essential concepts in Raft

* Raft works on a group of machines, or servers, by managing a replicated log. Applications are built on top of these Raft servers.
* There are in total 3 different state for Raft servers: **follower**, **candidate** and **leader**.
  * The job for **follower** is pretty simple, it responses to RPC call scheduled by **candidate** or **leader**.
  * The job for **candidate** is to initiate leader election by voting itself and collecting votes from other servers.
  * The job for **leader** is to replicate its log to **follower** and send heartbeats to maintain its **leader** state.
* Raft use a sequence of **term** to represent the entire process instead of using absolute time. Each **term** is uniquely identified by a number, which is monotonically increasing.
* Regarding to the lifetime of each **term**, they all start with *leader election*. Once a **leader** is elected, the **leader** repeatedly replicates its log onto all other servers, this process would continue as long as the **leader** could still send heartbeats to all **followers**. Otherwise, a new **term** would start.
* **Term** number is passed during each RPC request and reply. It is used to identify if the request is from staled servers and reject their requests. This information is also used by staled servers to reset their status to the right one.

For the leader election procedure, here are some highlights

* As shown in the figure, we would have a RequestVote RPC to do the job. Within the RequestVote handler, we would check if the request is a legit one by comparing term number, term number of last entry and index number of last entry. This part is covered in section 5.4.1. It is mainly used for checking which server is more up to date.
* If the request is smaller than the server, we would reject the vote. Otherwise, we would check if we have voted in the current term.
* Once we grant vote, we would need to reset the election timeout.
* Once election timeout is triggered. As long as the server is not **leader**, it will first convert itself to **candidate** state, vote for itself, reset election timeout and then start election.
* **Candidate** would send RequestVote RPC to all servers to collect votes in parallel and wait for response
* There are 3 potential outcomes for leader election:
  * The **candidate** collects majority of votes. It would convert itself to **leader** and start to send heartbeats to all others.
  * The **candidate** receives a heartbeat from other server that has **term** not smaller than its term, it would convert itself to **follower**.
  * The **candidates** could not collect enough votes and in this case the term ended with no **leader**. Election timeout would trigger and the **candidate** would start an election with a new term.
* The reason that **candidates** could not get majority is due to "vote splits". Raft use randomization to avoid this issue as much as possible. During the init and reset of election timeout, we would generate a new random number instead of using some fixed one.

**Leader** would periodically send heartbeats to each server. It uses the AppendEntries RPC to do the job. By sending RPC request with an empty log, it means that this is a heartbeats. Once a follower receive a heartbeat, it would reset its election timeout.

## Implementation

This is the first time that I work on concurrency programming and it is really challenging to get everything right. There are several suggestions on the [structure of Raft](https://pdos.csail.mit.edu/6.824/labs/raft-structure.txt) as well as how to [correctly use lock](https://pdos.csail.mit.edu/6.824/labs/raft-locking.txt), which is super beneficial on understanding Raft behavior and the design. Below are some hints that I think is critical to get Raft correct:

* There are several long-running processes in Raft, such as sending heartbeats and monitoring election timeout. It is better to use some long-running threads(goroutines in Go) to do these jobs.
* Since we need to send RPC in parallel, it is better to schedule the RPC calls within a thread and use a shared counters among these threads to collect the result.
* Since we are sending each RPC within separate threads, we don't know when would the thread going to be executed and what would be the status of the server at that time. So it is critical for us to check if the *server still matches our assumption when we step into the function*, such as if the term still matches or the state is still **candidates** etc.
* Avoid using lock around RPC calls, it is easy to cause dead lock.
* Use "print-to-console" style debugging is pretty efficient actually. Log the event out and see if the server is behaving abnormally.

The overall code is available [here](https://github.com/pyemma/mit-distributed-system/tree/master/src/raft).

### Leader election

For the leader election part, there are in total three pieces of work:

* RequestVote RPC handler
* Function to start election for **candidate**
* Background thread to monitor election timeout and start election

Here is the code of RequestVote RPC handler
{% highlight go %}
func (rf *Raft) RequestVote(args *RequestVoteArgs, reply *RequestVoteReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	DPrintf("%d receive request vote from %d on term %d, current term %d", rf.me, args.CandidateId, args.Term, rf.currentTerm)

	// if candidate term is smaller than current term, directly reject
	if args.Term < rf.currentTerm {
		reply.Term = rf.currentTerm
		reply.VoteGranted = false
		return
	}
	// if candidate term is larger than current term, convert to follower first
	if args.Term > rf.currentTerm {
		rf.convertToFollower(args.Term)
	}

	// not vote or the vote is the same as pervious candidate
	if rf.votedFor == -1 || rf.votedFor == args.CandidateId {
		// check if is more up to date
		if rf.isCandidateUpdateToDate(args) == false {
			reply.Term = rf.currentTerm
			reply.VoteGranted = false
			return
		}

		rf.votedFor = args.CandidateId // candidate is more up to date, vote for it
		rf.resetElectionTimer()        // whenever we make a vote, we need to reset the election timeout

		reply.Term = rf.currentTerm
		reply.VoteGranted = true
		return
	}

	reply.Term = rf.currentTerm
	reply.VoteGranted = false
	return
}
{% endhighlight %}

In the RequestVote handler, since we don't need to schedule any other RPC calls but change some shared variables, we acquire the lock from the entrance of the function. The first step is to check if the term in the request is outdated or not. If the term is outdated, then we directly reject the vote. Next step is to check if ourself is outdated or not. If our term is smaller, then we would immediately convert to **follower**. This has no effect for a **follower** but would make staled **candidate** change to the correct status. When covert to **follower**, we also reset the voted for variable. Although this step is not explicitly shown in the paper, this should be the correct behavior as we are in a new term and haven't vote for anyone yet.

Then we check if the **candidate's** logging is more up to date. This is described in section 5.4.1 as a restriction on the election. We put this part into a dedicated function below. The logic overall is exactly reproducing what the paper describes.

{% highlight go %}
// Check if the candidate is more up to date
func (rf *Raft) isCandidateUpdateToDate(args *RequestVoteArgs) bool {
	if len(rf.logEntries) > 0 { // there is some log entries on the server
		myLastLogEntry := rf.logEntries[len(rf.logEntries)-1]
		if myLastLogEntry.Term > args.LastLogTerm || (myLastLogEntry.Term == args.LastLogTerm && len(rf.logEntries)-1 > args.LastLogIndex) {
			return false
		}
	}
	return true
}
{% endhighlight %}

Once we have voted for the candidate, we need to reset the election timeout. Here is the piece of code doing this job

{% highlight go %}
// Reset election timer
func (rf *Raft) resetElectionTimer() {
	rf.electionTime = time.Now()
	rf.electionTimeout = time.Duration(rand.Intn(ElectionTimeOutRange)+ElectionTimeOutBase) * time.Millisecond
}
{% endhighlight %}

The way we implement timeout is to first record the start time and then regularly check if the current time minus the start time has exceed the timeout threshold or not. When we reset, we would use the current time to overwrite the start time, and assign a new timeout. This randomization is to help avoid split vote issues mentioned above.

Once we have finished the RequestVote RPC handler part (the sending RPC function is pretty simple and we would skip it), we need to implement the function that **candidates** use to start election. The entire logic is within the function `startElection`. The overall logic of it could be separate into three part: change status, send rpc and collect vote. Let's breakdown the function and look at it step by step.

The first part is to require the lock and update necessary status. The main logic is to change to **candidate**, increase term and vote for itself.
{% highlight go %}
rf.mu.Lock()
DPrintf("%d start election, pervious state %s", rf.me, rf.state)
// change to candidate and increase term
rf.currentTerm++
rf.state = Candidate
rf.votedFor = rf.me
rf.resetElectionTimer() // need to reset election timeout since we start a new election
// parameters to schedule RequestVote
term := rf.currentTerm
lastLogTerm := -1  // placeholder
lastLogIndex := -1 // placeholder
rf.mu.Unlock()
{% endhighlight %}

The next part is to send RPC calls to all servers. Since we are sending RPC calls here, we release the lock. We send each RPC within a goroutine. There are two important things to remember here:

* Information about the status of the current server such as term, lastLogTerm are passed into the goroutines as parameters. The reason is that at this particular timestamp the server matches our criteria and we need to create a snapshot to record it for later check due to the fact that other concurrency threads might change the status before the goroutines actually got scheduled.
* Similarly, we need to double check if the current status still matches our assumption when we start election. This is done by acquire the lock within the goroutine and compare the current status with the passed in status.
* We user 2 parameters to record the total number of RPC finished and the number of votes we collected. We also use a cond as a signal to wake up the thread that is waiting on some conditions to be satisfied.

{% highlight go %}
cond := sync.NewCond(&rf.mu)

count := 1 // counter to check the number of vote
finished := 1
// start send RequestVote in parallel
for peer := range rf.peers {
    if peer == rf.me {
        continue
    }

    go func(server int, term int, lastLogTerm int, lastLogIndex int) {
        rf.mu.Lock()
        // server is still candidate and current term matches the term when we plan to request vote
        if rf.state != Candidate || rf.currentTerm != term {
            rf.mu.Unlock()
            return
        }

        DPrintf("%d send request vote to %d on term %d", rf.me, server, rf.currentTerm)
        args := RequestVoteArgs{
            Term:         term,
            CandidateId:  rf.me,
            LastLogTerm:  lastLogTerm,
            LastLogIndex: lastLogIndex,
        }

        reply := RequestVoteReply{}
        rf.mu.Unlock()

        ok := rf.sendRequestVote(server, &args, &reply)

        rf.mu.Lock()
        defer rf.mu.Unlock()
        finished++
        if ok {
            DPrintf("%d got reply from %d on term %d with %t", rf.me, server, rf.currentTerm, reply.VoteGranted)
            if reply.Term > rf.currentTerm {
                rf.convertToFollower(reply.Term)
                return
            }
            // vote is granted, and the term matches the current term, and server is still candidate
            if reply.VoteGranted && reply.Term == rf.currentTerm && rf.state == Candidate {
                count++
            }
        }
        cond.Broadcast()

    }(peer, term, lastLogTerm, lastLogIndex)
}
{% endhighlight %}

The last part is to check if we have collected majority number of votes. If we haven't collect majority votes, or if the RPC call is not all finished yet, we would wait on the cond variable. This cond variable is also used in the goroutine we send RPC calls. The thread would be waked up in the goroutine when RPC call is finished, and the current thread would re-acquire the lock and check if the condition has matched or not. If we collect majority votes, we would convert to leader and start sending heartbeats.

{% highlight go %}
    rf.mu.Lock()
	// wait for enough vote or all request has returned
	n := len(rf.peers)
	majority := (n / 2) + 1
	for count < majority && finished != n {
		cond.Wait()
	}

	if count >= majority {
		DPrintf("%d collect majority of votes", rf.me)
		// change to leader and send the initial batch of heartbeats
		rf.state = Leader
		rf.sendHeartbeat()
		return
	}
	rf.mu.Unlock()
{% endhighlight %}

After we have a working function to start election, the next step, as well as the final part of leader election, is to have a background thread to periodically check the status of election timeout and trigger `startElection` at the right time. We check if the current state is not **leader** and if the time elapsed is greater than the timeout. If true then we start election, otherwise we would sleep 25ms and then check again.

{% highlight go %}
go func() {
    for {
        rf.mu.Lock()
        if rf.killed() {
            rf.mu.Unlock()
            return
        }
        // follower or candidate could start election upon election timeout
        if rf.state != Leader && time.Now().Sub(rf.electionTime) >= rf.electionTimeout {
            rf.mu.Unlock()
            rf.startElection()
        } else {
            rf.mu.Unlock()
            time.Sleep(25 * time.Millisecond)
        }
    }
}()
{% endhighlight %}

Above is all of the code related to leader election. Next, we would take a look at how heartbeats is implemented.

### Heartbeats

Compared with leader election, heartbeat is relatively simple. Heartbeat is implemented via AppendEntries RPC call. This RPC call is used by **leader** to send entires to **followers** to update their log. If the entries sent is empty, then it would be a heartbeat signal, which its pure function is to reset **follower's** election timeout so that the **leader** could maintain its status.

Similar to leader election, the work is also separated into 3 parts: AppendEntires RPC handler, function to send heartbeats and background thread to repeatedly send heartbeats.

The AppendEntries RPC handler is pretty straight forward. It checks if the request is legal (a.k.a request term is not smaller than its current term) and covert itself to follower if its term is outdated. And then reset the election timeout.

{% highlight go %}
func (rf *Raft) AppendEntries(args *AppendEntriesArgs, reply *AppendEntriesReply) {
	// This is a heartbeat
	if len(args.Entries) == 0 {
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

		reply.Term = rf.currentTerm
		reply.Success = true
		return
	}
}
{% endhighlight %}

The sending heartbeats part is also a simplified version of request votes and don't have too much to talk about. The background thread to periodically send heartbeats shares similar structure with the one to check election timeout.

{% highlight go %}
go func() {
    for {
        rf.mu.Lock()
        if rf.killed() {
            rf.mu.Unlock()
            return
        }
        if rf.state == Leader && time.Now().Sub(rf.heartbeatTime) >= HeartbeatTimeout {
            rf.sendHeartbeat()
        } else {
            rf.mu.Unlock()
            time.Sleep(10 * time.Millisecond)
        }
    }
}()
{% endhighlight %}

## Future work

In the following posts, we would implement the AppendEntires RPC call to be able to replicate **leader's** log onto **followers**.