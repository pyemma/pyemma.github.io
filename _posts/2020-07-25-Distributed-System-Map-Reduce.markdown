---
layout: single
title: MIT Distributed System Course - MapReduce
tags:
- distributed system
---

## Introduction

This is going to be a series of posts to record my learning of [MIT 6.824 Distributed System](https://pdos.csail.mit.edu/6.824/schedule.html). The post would focus on the course assignments which is to build some distributed systems from scratch using [Go language](https://golang.org/). I would discuss some of the basic ideas that these assignments touched, as well as the reason I reached to the design decision of my implementation. For the code, here is some disclaimers:
* The implementation would guaratee to pass the test script provided by the course
* The implementation might not be in the most optimal status, I might have follow up posts to refactor the implementation later
* The implementation might not be in the cleanest way, as I'm still learning Go language

## MapReduce
MapReduce has been known for a long time as a framework to handle large scale data processing jobs. ***The first assignment is to build a simple version of MapReduce to do word count (a classic demo application for MapReduce xD)***.

The original [MapReduce paper](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf) includes lots of details about how it works overall and some tricks Google applied in their implementation. Here is a summary of some key points.

There are two types of machines in the system:
* Master: Coordinate the entire process, schedule Map/Reduce task to workers, store necessary metadata to track the progress of the process
* Worker: Do the actual Map/Reduce task with users' program

and there are two types of task:

* Map: Read a split of data assigned and pass it to users' map functions, generate intermediate key/value paris and store them in local files, propagate intermediate files to master
* Reduce: Read all intermediate files according to some partition functions which distribute key/value pairs to different intermediate files, sort key/value pairs and pass to users' reduce function, then output to final files

There are some tricks mentioned in the paper to improve the performance:
* Locality, Map task is scheduled to the machine where the input splits stored. This helps save the precious network bandwidth.
* Backup task is scheduled to resolve the straggler issue, which is the case that some workers are making progress unexpected slow.
* Heartbeat is used by master to check the status of workers
* If a worker failed, then all the Map task completed by the worker would be reset to idle. The Map task and Reduce task in progress would also be reset to idle.

## Implementation

The full implementation is available [here](https://github.com/pyemma/mit-distributed-system/tree/master/src/mr). 

### RPC
In this assignment, master and worker communicate via RPC calls. For example, a worker schedules a RPC call to master to get a task to work, and notifies master that the task has been finished via another RPC call. Here is a list of RPC in my implementation:
{% highlight go %}
func (m *Master) RegisterWorkerId(args *RegisterWorkerIdArgs, reply *RegisterWorkerIdReply) error

func (m *Master) RequestJob(args *RequestJobArgs, reply *RequestJobReply) error

func (m *Master) FinishJob(args *FinishJobArgs, reply *FinishJobReply) error

{% endhighlight %}

* RegisterWorkerId is the first RPC call each worker would schedule once up. This call is to register itself within master's in memory status tracker so that master is aware of its existence and assigns it a unique identifer. The reason of this behavior is that master might need to reschedule a task to other workers if the original worker is a straggler. And the master need to use this identifier to check if the result is returned from the new worker instead of the staled work for the sake of security.

* RequestJob is a RPC call that workers call repeatedly. Master would assign a task for it if there is any task could be schedule to this worker or reply there is no available task. Once all tasks are done, a dedicated status would be returned to workers so that they could safely exits.

* FinishJob is a RPC call that workers call once the assigned work is done. Worker would also pass necessary information via this RPC call to master.

### Master Record Data Structure

As master is the coordinator for all workers, it is critical for master to keep track of all workers' progress. In this assignment, I highlight the following points for master to take care of:
* Workers that are registered within the system and their identifiers.
* Map/Reduce task identifiers. I made a simplification that, number of input files is tantamount to the number of map tasks. The number of reduce tasks is a parameter passed in once master is up.
* Map/Reduce task status. There are in total 3 status: Not Scheduled/Pending/Done. If a task is in pending status, we would also record the worker id that is assigned with this task. This information is critical because the worker might fail or be straggle on the task. To avoid infinite waiting, master has a timeout of 10s on each task scheduled. So there might be cases that, some workers are straggle and would respond after 10s when master has already assigned the task to another worker. We would like to reject the response by this straggler worker. So only when the worker identifier matches master's record, the response would be accepted.
* Map task input filenames and intermediate filenames.
* 2 boolean parameters to check if all map tasks are done and all reduce tasks are done.
* A mutex is used to lock the master data structure. The reason is that Go process each RPC call via treads and each tread is sharing the same copy of the record. To avoid racing condition, each thread needs to first acquire the lock and then does the necessary update.

Below is a piece of the code

{% highlight go %}
type Master struct {
	// need to lock the object in concurrent env
	mu sync.Mutex
	// use an array to record all workers, we need to use this to reject
	// reply from workers that dead or straggle
	workers []int
	// use a array here to store the mapper input filenames, this is
	// used to generate the mapper task id
	mapperFilenames []string
	// use a dict here to store the status of mapper job
	mapperStatus map[int]JobStatus
	// use a map to store reducer input filenames
	reducerFilenames map[int][]string
	// use a dict here to store the status of reducer job
	reducerStatus  map[int]JobStatus
	numReducer     int
	mapperAllDone  bool
	reducerAllDone bool
}
{% endhighlight %}

### Master

Master implements the three RPC APIs mentioned in [RPC](#rpc). I would skip RegisterWorkerId as it is pretty straight forward to implement and mainly focus on the RequestJob and FinishJob RPC call.

> ***RequestJob***

For RequestJob, all map tasks need to be done before the scheduling of any reduce task. The master would first lock the record, and see if there is any map task available. Once all map tasks are done, the corresponding flag would be enabled, and master would check if there is any available reduce task. Master would reply with a data type RequestJobReply.

{% highlight go %}
type RequestJobReply struct {
	JobType    string   // map/reduce job
	JobId      int      // map/reduce job id
	Filenames  []string // the input filename to map/reduce job
	NumReducer int      // number of reduce tasks in total
}
{% endhighlight %}

* JobType is to tell worker this is Map task or Reduce task.
* JobId is the identifier assigned by master to track the job's status.
* Filenames could be the map input filenames or intermediate filenames generated by worker.
* NumReducer is required to tell how many intermediate files we need to generate and shard the keys accordingly.

> ***FinishJob***

For FinishJob, master would read the result generated by workers and update the corresponding record. The logic of reduce task is pretty simple. For map task, master needs to parse the returned filenames and figures out the target reduce task id. For all map/reduce task, we need to check if the status is Pending and the worker id on record matches the worker id in the request before updating the record.

Here is the structure of the FinishJob request, which is constructed by worker and send to master:
{% highlight go %}
type FinishJobArgs struct {
	JobType   string
	JobId     int
	WorkerId  int      // worker id
	Filenames []string // filenames that are generated
}
{% endhighlight %}

Here is the logic of parsing filenames and updating the corresponding record:
{% highlight go %}
case Map:
    // Mark the corresponding Map job finished, and flash the intermediate file
    jobId := args.JobId
    filenames := args.Filenames
    // There might be duplicated job due to straggler, check the status to see if
    // the reply matches our record
    jobStatus, _ := m.mapperStatus[jobId]
    if jobStatus.Status == Pending && jobStatus.WorkerId == args.WorkerId {
        m.mapperStatus[jobId] = JobStatus{Status: Done, WorkerId: -1}
        for _, filename := range filenames {
            token := strings.Split(filename, "-")
            reducerId, _ := strconv.ParseInt(token[len(token)-1], 10, 64)
            m.reducerFilenames[int(reducerId)] = append(m.reducerFilenames[int(reducerId)], filename)
        }
    }
{% endhighlight %}

> ***CheckStatus***

As mentioned in [Master Record Data Structure](#master-record-data-structure), master has a timeout of 10s on each task scheduled. Once the task has been timeout, master would put the task back to Not Scheduled status so that it could be allocated to other workers, and clear the worker id on the record. This is achieved by firing a goroutines of func CheckStatus each time we schedule a task to a worker. Function CheckStatus would first sleep 10s and then check the status of the task. If the task is still in Pending status, then it would put it back to Not Scheduled status, otherwise it would directly return. 

{% highlight go %}
func (m *Master) CheckStatus(jobType string, jobId int) {
	time.Sleep(time.Second * 10)
	m.mu.Lock()
	defer m.mu.Unlock()
	if jobType == Map {
		jobStatus := m.mapperStatus[jobId]
		if jobStatus.Status != Done {
			m.mapperStatus[jobId] = JobStatus{Status: NotScheduled}
		}
	} else {
		jobStatus := m.reducerStatus[jobId]
		if jobStatus.Status != Done {
			m.reducerStatus[jobId] = JobStatus{Status: NotScheduled}
		}
	}
}
{% endhighlight %}

### Worker

Worker takes user's customization map and reduce function as input. In this assignment, this is done via Go plug-ins. Once a worker is up, the first thing it tries to do is to call the RegisterWorkerId RPC to let the master know its existence and assign it an identifier. Each worker would have a unique identifier and this is guaranteed by the master. It then enters an infinite loop. Within the loop, the worker would try to first get a task from the master. Once it gets a map task:
* Read in the file content
* Pass all content into mapf function, which is the users' map function
* Shard the output according to the num of reduce tasks
* Write all content into a temp file and then atomically rename it following the convention of `out-X-Y`, where X is the map task id and Y is the reduce task id
    * The reason that we write into a temp file is that we don't want to expose any partially written file. Only when all content is flushed into the file, we would formally rename it to the acceptable intermediate filename.
* Call FinishJob with the map task id, worker id and all intermediate filenames generated to the master
{% highlight go %}
case Map:
    filename := reply.Filenames[0]
    file, _ := os.Open(filename)
    content, _ := ioutil.ReadAll(file)
    file.Close()
    kva := mapf(filename, string(content))

    // shard the kv into nReducer splits
    kvSharded := make(map[int][]KeyValue)
    for _, kv := range kva {
        shard := ihash(kv.Key) % numReducer
        kvSharded[shard] = append(kvSharded[shard], kv)
    }
    // Flash the result to files
    for shard, kva := range kvSharded {
        tempfile, _ := ioutil.TempFile("", "mr-tempfile")
        // Use json format to store the result in intermeidate file
        enc := json.NewEncoder(tempfile)
        enc.Encode(&kva)
        tempfile.Close()
        interFilename := "mr-" + strconv.Itoa(reply.JobId) + "-" + strconv.Itoa(shard)  // mr-X-Y
        os.Rename(tempfile.Name(), interFilename)
        intermediate = append(intermediate, interFilename)
    }
    FinishJob(workerId, reply.JobType, reply.JobId, intermediate)
{% endhighlight %}

Once it gets a reduce task:
* Read all intermediate files content
* Sort all keys
* Pass to reducef. This part of logic is provided by the assignment example 
* Flush result into temp file and once everything is done rename it to `out-X` where X is the reduce task id

{% highlight go %}
case Reduce:
    // Read values from all intermediate file
    kva := make([]KeyValue, 0)
    for _, filename := range reply.Filenames {
        file, _ := os.Open(filename)
        dec := json.NewDecoder(file)
        kvs := make([]KeyValue, 0)
        if err := dec.Decode(&kvs); err != nil {
            log.Fatalf("fail to read intermediate file")
        }
        kva = append(kva, kvs...)
    }

    sort.Sort(ByKey(kva))

    oname := "mr-out-" + strconv.Itoa(reply.JobId)
    tempfile, _ := ioutil.TempFile("", "mr-tempfile")
    // This part of code is copied from the course example code
    i := 0
    for i < len(kva) {
        j := i + 1
        for j < len(kva) && kva[j].Key == kva[i].Key {
            j++
        }
        values := []string{}
        for k := i; k < j; k++ {
            values = append(values, kva[k].Value)
        }
        output := reducef(kva[i].Key, values)

        fmt.Fprintf(tempfile, "%v %v\n", kva[i].Key, output)

        i = j
    }
    tempfile.Close()
    os.Rename(tempfile.Name(), oname)
    intermediate = append(intermediate, oname)

    FinishJob(workerId, reply.JobType, reply.JobId, intermediate)
{% endhighlight %}

Once the worker receives None flag from RequestJob response, it would sleep for 2s and then continues. Once it receives AllDone, it would break the loop and exit.

## Future Plan

This is just a naive implementation and there are several points that could be optimized.
* The master data structure could be simplified, e.g.
    * Map/Reduce task status could be merged into a single one
    * We could keep a queue to check if there are still map/reduce tasks not scheduled yet
* The logic within worker's map/reduce processing could be optimized, e.g.
    * When we write and read intermediate file, we are processing sequentially. But we could actually do this concurrently by splitting the work of each intermediate file into a goroutine

Please leave any comment and feedback you have :)