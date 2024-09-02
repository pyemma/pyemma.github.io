---
layout: single
title: LLM Training 101
tags:
- llm
- llm training
- distributed training
toc: true
---

这个是读完这篇综述 [Efficient Training of Large Language Models on Distributed Infrastructures - A Survey](https://arxiv.org/pdf/2407.20018) 之后的一个产出，这篇综述文章针对 LLM 的 training 介绍的已经很详细了，但是同时内容过多也不可能全都学完。这里针对自己整理的一些笔记来列一个之后学习的提纲，这个提纲肯定是非常主观的，推荐大家去读读原文来根据自己的情况针对性的准备

> PS: 后续会不定期的更新这篇 blog 来争取与时俱进，同时会有专栏来介绍这篇 blog 里面打算深入研究的项目

## 概念性知识

- LLM 训练的一些特点
  - 模型架构的一致性，基本都是堆的 transformer, 虽然现在有一些不一样的尝试比如 Mamba 和 TTT, 但是主流的模型还是 transformer
  - 训练的规模和时间也是空前绝后的
  - Specialized software, 比如 Megatron (这个听说过，去了解一下)
  - LLM 训练的 pipeline 也发生了变化（这一点说的还蛮有道理，我在这个领域有比较多的经验，可以向这个 LLM 的方向研究一下看看有什么机会）。传统的机器学习都是针对某一个问题用对应的数据来训练（domain specific），但是现在 LLM 的主流是在大量的数据做自监督学习，然后再进行 fine-tuning, alignment 等
  - 在 LLM 训练的各项因素之中，Communication overhead 是一个主要痛点
- LLM 训练的 infrastructure 相关的内容
  - PCIe 由于 bandwidth 的问题导致其不是很合适 LLM 的训练，现在更多的是使用专用的链接比如 NVLink 等，同时能使用不同的网络连接拓扑结构来进行进一步的优化，比如 cube-mesh 或者 switch-based fullly-connected
  - The Clos network architecture, commonly known as a Fat-Tree topology, is widely used in LLM training clusters. In a Closbased cluster, each server, equipped with one or more NICs, is organized into racks connected to leaf switches. These leaf switches link to spine switches, providing inter-rack connectivity and forming a pod. The pods are further interconnected with core switches, facilitating any-to-any communication across servers within the cluster.
  - Parallel file systems such as Lustre, GPFS, and BeeGFS are frequently deployed on leading high performance computing systems to ensure efficient I/O, persistent storage, and scalable performance. 听说过 distributed file system, 但是这个 parallel file system 是啥

## 打算去学习了解的框架和技术

- **RDMA**: 可以去学习了解一下 InfiniBand
- **DeepSpeed-Chat**, parallel strategy
  - <https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat>
  - uses Hybrid Engine to seamlessly switch model partitioning between training and inference, such as using tensor parallelism to improve throughput during inference and using ZeRO or LoRA to improve memory utilization during training, providing outstanding system efficiency for RLHF training
- **HuggingFace TRL**, parallel strategy
  - <https://huggingface.co/docs/trl/en/index>
  - make full use of various parameter-efficient fine-tuning (PEFT) methods, such as LoRA or QLoRA, to save memory cost, and use a dedicated kernel designed by unsloth to increase the training speed of RLHF.
- **FlashAttention**, 内存优化
  - <https://github.com/Dao-AILab/flash-attention>
  - an IO-aware tiling algorithm is proposed to reduce the number of memory reads/writes between slow HBM and fast on-chip SRAM based on the online softmax. 看能不能自己实现一遍这个算法，网上应该有一些简化版的 kernel 教程，可以参考学习一下
  - Selective-checkpointing selectively discards the activations of memory-intensive attention modules. FlashAttention fuses the attention module into a single kernel, and also employs selective-checkpointing to reduce memory consumption. 这个看一下具体是怎么做的
- **FlashAttention 2**: 内存优化, efficiently handles variable-length inputs by parallelizing the sequence length dimension inseparably
  - 这个是怎么实现的，去学习一下代码
- **FlashAttention 3**: 内存优化, An interleaved block-wise GEMM and softmax algorithm is redesigned based on FlashAttention-2 to hide the non-GEMM operations in softmax with the asynchronous WGMMA instructions for GEMM. Besides, by leveraging the asynchrony of the Tensor Cores and Tensor Memory Accelerator (TMA), overall computation is overlapped with data movement via a warp-specialized software pipelining scheme. Blockwise Parallel Transformer (BPT) further reduces the substantial memory requirements by extending the tiling algorithm in FlashAttention to fuse the feedforward network
  - 需要学习了解一下 WGMMA, Tensor Cores, Tensor Memory Accelerator, Blockwise Parallel Transformer
- **Triton**, 用来写 kernel, 计算优化，听说现在很多公司内部在大量的使用这个写 Kernel, 可以学习一下 #kernel #CUDA
  - <https://github.com/triton-lang/triton>
- **ZERO**, 通过 fully sharding 来进行内存优化, ZERO1, 2, 3
  - <https://arxiv.org/pdf/1910.02054>
  - ZeRO-3 employs per-parameter sharding to shard the full model and utilizes All-Gather and ReduceScatter for unsharding and sharding communication, respectively
  - [**ZERO++**](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/) 感觉也算是 ZERO 家族的一员，但是是一种 partial sharding 的办法，在 ZERO3 的基础之上, further introduces a secondary shard of parameters within subgroups of GPUs and uses quantization to compress parameters and gradients, effectively diminishing communication volume with a trade-off in accuracy
  - [**ZeRO-Offload**](https://arxiv.org/pdf/2101.06840) concentrates on multi-GPU training. It holds model parameters on GPU, and stores optimizer states and gradients on CPU memory. In addition, it offloads optimizer update computation to the CPU.
- **Ring AllReduce** 算法: <https://github.com/baidu-research/baidu-allreduce>
- [**Horovod**](https://arxiv.org/pdf/1802.05799): replaced the Baidu ring-AllReduce implementation with NCCL and designed a user-friendly interface for distributed training
- **Pytorch DPP**: fuse multiple sequential AllReduce communication operations into a single operation. This method avoids transmitting a large number of small tensors over the network by waiting for a short period of time and then combining multiple gradients into one AllReduce operation during the backward phase. 通信优化的一种办法，可以看看代码学习一下
- **FSDP**: <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
- [**GPipe**](https://arxiv.org/pdf/1811.06965) 是之前听说过的一种方法，貌似是目前比较流行的方法，但是仍然会在开始和结束的时候有大量的 bubble 出现
  - <https://github.com/kakaobrain/torchgpipe>

## 一些比较主流的和重要的概念

### Parallelism Strategy

- **Tensor parallelism**: partitions the parameter tensors of each layer along multiple dimensions, effectively distributing the model parameters across the available GPUs. 感觉 tensor parallelism 没有 data/model parallelism 那么常见，在工作中没怎么看到用这种方法的
  - it is challenging to overlap the communication with computation, necessitating the use of high-bandwidth connections. Consequently, tensor parallelism is more commonly employed in a single GPU node.
- **Pipeline parallelism**: pipeline parallelism only necessitates the exchange of intermediate tensors at designated cutting points, resulting in less frequent communication requirements, pipeline parallelism 算是比较常用的东西了
  - 但是 pipeline parallelism 也有两个问题，一个是 pipeline bubble, 一个是 memory consumption imbalance
- **Sequence parallelism**: It divides the input data into multiple chunks along the sequence dimension and each chunk is fed to one GPU for computation. 没怎么听说过这种方法，可以找来一些 code 来学习一下
  - [MQA](https://arxiv.org/pdf/1911.02150) 和 [GQA](https://arxiv.org/pdf/2305.13245) 就是属于这个范畴, 可以好好的学习一下
  - [**Ring Self-Attention**](https://arxiv.org/abs/2310.01889) leverages sequence parallelism and calculates the self-attention with ring-style communication to scale up the context window of LLM training. It first transmits the key tensors among GPUs to calculate the attention scores in a circular fashion, and then calculates the self-attention output based on the attention scores and value tensors transmitted in a similar fashion
- **MoE parallelism**: MoE 的结构在目前主流的 LLM 里面都得到了大量的使用，可以看看下面的这几篇文章里面介绍的针对 MOE 的 parallel strategy 的方法 #MOE
  - [**GShard**](https://arxiv.org/abs/2006.16668): extends the idea of MoE to Transformers in distributed settings, where experts are distributed across different workers and collaborates with All-to-All communication
  - [**DeepSpeed-MOE**](https://arxiv.org/abs/2201.05596): proposes a new distributed MoE architecture that applies shared experts in each worker and places more experts in deeper layers to balance communication costs with training accuracy
- Since General Matrix Multiplications (GeMMs) require the size of all experts’ inputs to be consistent, existing MoE training frameworks often perform token dropping and padding to match the same expert capacity, which wastes computation.
  - General Matrix Multiplications (GeMMs) 的工作原理可以参考: <https://spatial-lang.org/gemm>
  - Token dropping and padding 的常用方法是什么？有没有具体的实现代码样例
- 针对 MOE parallel strategy 中 communication 的优化
  - [**Tutel**](https://arxiv.org/abs/2206.03382): divides the input tensors into groups along the expert capacity dimension and overlaps computation and communication among different groups to hide All-to-All overhead
  - **Tutel**: optimizes the All-to-All kernel implementation by aggregating small messages into a single large chunk inside the nodes before exchanging data among different nodes #Batching
  - [**Lina**](https://www.usenix.org/system/files/atc23-li-jiamin.pdf) analyzes the All-to-All overhead of MoE during distributed training and inference systematically and finds that All-to-All latency is prolonged when it overlaps with AllReduce operations. Lina proposes prioritizing All-to-All over AllReduce to improve its bandwidth and reduce its blocking period in distributed training
    - 很有意思的发现，可以去学习一下原文里面是怎么发现这个问题的，然后应用在自己以后的工作中
- This heterogeneity is also reflected in model architectures, particularly with Reinforcement Learning from Human Feedback (RLHF). Utilizing heterogeneous hardware and diverse model architectures has become essential for the efficient training of LLMs
  - 再重新学习一下 RLHF，来理解这里面提到的 **异构性** 的特点

### Memory Optimization

- [Rabe](https://arxiv.org/abs/2112.05682) 这篇论文中证明了自注意力只需要 O(logn) 的内存就可以了，学习一下这篇论文里面的工作
- 了解一下 FP16 和 BF16 的工作原理，内存优化
- LLM training 的过程中主要吃内存的部分
  - Model States: Model states encompass the memory consumed by the optimizer states, gradients, and model parameters
  - Activations refer to the tensors generated during the forward pass
  - Temporary Buffers: Temporary buffers are used to store intermediate results
  - Memory Fragmentation: Memory fragmentation can lead to scenarios where memory requests fail despite having a large amount of available memory, 这个在 Pytorch 里面由于内存分配机制会出现这种问题，可以再找一些额外的资料详细的了解一下
- Deep learning frameworks typically use a caching allocator with a memory pool to enable fast memory allocation and deallocation without requiring device synchronization.
- 一个用来估算所需要的内存的简易办法
  - When training a model with Φ parameters,4Φ bytes are needed to store parameters and gradients. The 32-bit copies of the parameters, momentum, and variance each require 4Φ bytes, totaling12Φ bytes. Therefore, the overall memory requirement for storing model states is 16Φ bytes，这个再好好看一下理解一下
- 一些用来进行 Memory 优化的整体大方向
  - Activation re-computation strategies, which trade increased computation for reduced memory usage, 这个是现在最主流的方法之一，可以找一些代码来看看是如何实现的，这个方法的一个关键就是节省的内存和额外计算之间的 trade off
  - Redundancy reduction methods that minimize data duplication across training processes
  - Defragmentation techniques that optimize memory allocation and deallocation to reduce fragmentation and improve memory utilization
- [**GMLake**](https://arxiv.org/abs/2401.08156) and [**PyTorch expandable segments**](https://pytorch.org/docs/stable/notes/cuda.html) propose to mitigate fragmentation by utilizing the virtual memory management (VMM) functions of the low-level CUDA driver application programming interface. 可以看看 PyTorch 里面这个工作
- Swap and offload approaches that leverage CPU memory and NVMe SSDs to supplement GPU memory
  - CPU offloading: static/dynamic
  - SSD offloading, 这个在之前的 GPU training paper 里面好像看到过

### Communication Optimization

一些和通信相关的优化

- NVIDIA’s NCCL and AMD’s RCCL are highly optimized libraries that typically outperform MPI-based collective communication libraries on their respective AI accelerators. These libraries usually select pre-defined algorithms to perform collectives based on conditions such as network topology and input tensor size. 可以去学习一下 NCCL
- 通信的不同算法: **Ring, Tree, Hybrid**
- Conventional frameworks simultaneously perform gradient computation for both weights and outputs. [**Out-of-order backpropagation (ooo-backprop)**](https://github.com/mlsys-seo/ooo-backprop) decouples the gradient computations for weights and outputs, scheduling the weight gradient computations flexibly out of their original order. This allows more critical computations to be prioritized and scheduled accordingly. Consequently, ooo-backprop optimizes overall performance by scheduling communications based on this out-of-order computation strategy. 这个工作很有意思，把 activation 和 gradient 的 communication 拆开然后进行类似不同的 priority 的 communication
- **In-network aggregation (INA)** uses the computational capabilities of network devices to perform aggregation operations like summing gradients of deep learning models.

### Fault Tolerance

Failure tolerance 主流的还是使用 checkpoint

- **Synchronous checkpoint**
- [**Check-N-Run**](https://www.usenix.org/system/files/nsdi22-paper-eisenman.pdf) decouples the snapshot and persist phases. It achieves atomic checkpointing by stalling training only during the snapshot phase and asynchronously persisting snapshots using dedicated background CPU processes.
- [**DeepFreeze**](https://web.cels.anl.gov/~woz/papers/DeepFreeze_2020.pdf) applies both lightweight (snapshot) and heavy(persist) persistence strategies in the background, sharding checkpoints across data-parallel GPUs to distribute I/O workload.
- **Gemini** proposes checkpointing to CPU memory for faster failure recovery, along with a checkpoint placement strategy to minimize checkpoint loss and a traffic scheduling algorithm to reduce interference with training.
- [**Tectonic**](https://www.usenix.org/system/files/fast21-pan.pdf): Meta’s distributed filesystem, enables thousands of GPUs to save and load model checkpoints simultaneously, providing efficient and scalable storage solutions for extensive training operations
- 现在貌似主要用来对 checkpoint 用来存储的都是 object store, 这个可以去研究下看看各个公司都用啥（比如 AWS 是不是都上 S3）
- Live migration leverages the inherent redundancy present in distributed LLM training setups, particularly the model replicas across different data parallel pipelines, to restore model states in case of failure. 这个感觉其实有点类似使用 Cassandra 里 consistency hashing 里面的 hinted hand off