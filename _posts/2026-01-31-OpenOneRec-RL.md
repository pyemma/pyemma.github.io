---
title: Learning VERL Part 1 - A Perspective from OpenOneRec
date: 2026-01-31
author: pyemma
categories: [Machine Learning]
tags: [verl, rl-infra]
toc: true
math: true
---

This post documents my journey learning [VERL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning), a scalable and efficient reinforcement learning framework, through the lens of [OpenOneRec](https://github.com/pyemma/OpenOneRec)'s implementation. OpenOneRec uses VERL to implement PPO-based training for recommendation systems with a two-stage generation approach (Chain-of-Thought reasoning followed by item ID generation).

VERL provides a sophisticated abstraction layer over Ray for distributed RL training, handling complex orchestration between actor policies, reference policies, critics, and rollout workers. This analysis focuses on how OpenOneRec leverages VERL's infrastructure to implement its reinforcement learning pipeline.

The introduction in this post is relative high level. I plan to dive deeper into the VERL framework and share more learnings in the future blogs.

## High-Level Architecture Overview

The overall PPO training flow in OpenOneRec consists of several key components working together:

### Main Components and Their Roles

```bash
┌─────────────────────────────────────────────────────────────────┐
│                     main_onerec_ppo.py                          │
│                     (Entry Point)                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OneRecTaskRunner                             │
│  • Loads configuration                                          │
│  • Creates resource pools and dataloaders                       │
│  • Initializes worker groups and trainer                        │
└─────┬─────────────────┬─────────────────┬────────────────┬──────┘
      │                 │                 │                │
      │ creates         │ configures      │ configures     │ configures
      ▼                 ▼                 ▼                ▼
┌─────────────┐   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│RayPPOTrainer│   │Actor/Rollout/│  │   Critic     │  │   Reward     │
│             │   │  Ref Worker  │  │WorkerGroup   │  │WorkerGroup   │
│             │   │  WorkerGroup │  │              │  │              │
└──────┬──────┘   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       │ orchestrates    │                 │                 │
       └─────────────────┼─────────────────┼─────────────────┘
                         │
                         │ implements
                         ▼
         ┌───────────────────────────────────────┐
         │  OneRecActorRolloutRefWorker          │
         │  (Hybrid: Actor/Rollout/Reference)    │
         └───┬───────────┬───────────┬───────────┘
             │           │           │
             │ uses      │ uses      │ uses
             ▼           ▼           ▼
    ┌────────────┐  ┌─────────────────────┐  ┌──────────────────┐
    │OneRecvLLM  │  │FSDPVLLMShardingMgr  │  │DataParallelPPO   │
    │Rollout     │  │                     │  │Actor             │
    │(2-stage)   │  │(Param sync)         │  │(Training)        │
    └────────────┘  └─────────────────────┘  └──────────────────┘
```

1. **OneRecTaskRunner**: The orchestrator that initializes the entire training pipeline. It:
   - Loads configuration and identifies the appropriate `ActorRolloutRefWorker` implementation
   - Creates `RayWorkerGroup` instances for different roles (actor, critic, reward)
   - Initializes `ResourcePoolManager` for GPU allocation and scheduling
   - Sets up the dataloader for training batches
   - Creates `RayPPOTrainer` and invokes the training entry point

2. **RayPPOTrainer**: The coordinator for the PPO algorithm. It:
   - Manages the overall scheduling of the PPO algorithm
   - Delegates specific responsibilities (`Actor`, `Critic`, `Ref`) to worker classes
   - Initializes worker groups based on resource pool specifications
   - Maps roles to resources using `RayClassWithInitArgs`
   - Optimizes resource usage through colocation (multiple roles can share the same resource pool)

3. **RayWorkerGroup**: The distributed execution engine. It:
   - Takes `RayResourcePool` and `RayClassWithInitArgs` during initialization
   - Binds worker class methods to the WorkerGroup using `_bind_worker_method`
   - Routes method calls (e.g., `generate_sequence`) to the appropriate worker implementation
   - Handles distributed communication patterns (similar to FSDP2's parameter sharding/unsharding via dispathers and `@register`)

4. **OneRecActorRolloutRefWorker**: OpenOneRec's customized worker implementation that:
   - Overrides `_build_rollout` to use the custom `OneRecvLLMRollout`
   - Integrates FSDP2 for distributed training and vLLM for efficient inference

5. **OneRecvLLMRollout**: Implements the two-stage rollout:
   - Stage 1: Generate Chain-of-Thought (CoT) reasoning
   - Stage 2: Generate item ID sequences using beam search

### Training Loop Data Flow

```bash
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────┐
│ DataLoader  │     │ RayPPOTrainer    │     │Actor Worker   │     │Critic    │
│             │     │                  │     │Group          │     │Worker    │
└──────┬──────┘     └────────┬─────────┘     └───────┬───────┘     └────┬─────┘
       │                     │                       │                   │
       │ For each epoch:     │                       │                   │
       │                     │                       │                   │
  1.   │──── batch ─────────>│                       │                   │
       │                     │                       │                   │
  2.   │                     │── generate_sequence ─>│                   │
       │                     │                       │                   │
  3.   │                     │<── sequences ─────────│                   │
       │                     │    (with old_log_probs)                   │
       │                     │                       │                   │
  4.   │                     │─ compute_reward()     │                   │
       │                     │                       │                   │
  5.   │                     │─ compute_ref_log_prob ─>                  │
       │                     │<─ ref_log_probs ──────│                   │
       │                     │                       │                   │
  6.   │                     │─ compute_advantage()  │                   │
       │                     │                       │                   │
  7.   │                     │─────────── update_critic ────────────────>│
       │                     │<────────── critic_metrics ────────────────│
       │                     │                       │                   │
  8.   │                     │─── update_actor ──────>│                   │
       │                     │    (after warmup)     │                   │
       │                     │<── actor_metrics ─────│                   │
       │                     │                       │                   │
       │                     │ (repeat for next batch)                   │
       ▼                     ▼                       ▼                   ▼
```

The `fit` method in `RayPPOTrainer` orchestrates this flow:

- **Generate sequences**: Call `actor_rollout_wg.generate_sequence(batch)`
- **Compute rewards**: Apply reward function to generated sequences
- **Compute log probabilities**: Get old policy and reference policy log probs
- **Compute advantages**: Calculate PPO advantages using GAE or other estimators
- **Update critic**: Train value function (if using critic)
- **Update actor**: Perform PPO policy update after critic warmup

This architecture enables efficient distributed RL training by separating concerns, optimizing resource usage, and providing clean abstractions for extending the system.

## Fundamental Building Blocks

Before diving into specific components, it's essential to understand two fundamental building blocks that power VERL's distributed execution: `RayWorkerGroup` and the `@register` decorator. These abstractions make distributed RL training manageable by hiding complex coordination logic.

### RayWorkerGroup: Abstraction Over Ray for Distributed RL

While Ray provides powerful primitives (`@ray.remote`, `ray.get()`, `Actor`), distributed RL training has specific requirements that would require significant boilerplate code. `RayWorkerGroup` provides a clean abstraction layer that handles:

#### 1. Automatic Distributed Training Coordination

**Without RayWorkerGroup**, you'd need to manually manage:

- MASTER_ADDR, MASTER_PORT discovery
- RANK assignment for each worker
- WORLD_SIZE propagation

**With RayWorkerGroup**:

```python
# Automatically handles MASTER_ADDR, MASTER_PORT, RANK assignment
actor_worker_group = RayWorkerGroup(
    resource_pool=RayResourcePool(process_on_nodes=[8]),
    ray_cls_with_init=RayClassWithInitArgs(ActorWorker, config=config)
)

# Each worker automatically gets environment variables:
# - RANK (0-7), WORLD_SIZE (8)
# - MASTER_ADDR, MASTER_PORT (from register center)
# - LOCAL_RANK, LOCAL_WORLD_SIZE
```

The framework automatically injects these environment variables:

```python
env_vars = {
    "WORLD_SIZE": str(world_size),
    "RANK": str(rank),
    "WG_PREFIX": self.name_prefix,
    "WG_BACKEND": "ray",
    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
    "RAY_LOCAL_RANK": str(local_rank),
}
if rank != 0:
    env_vars["MASTER_ADDR"] = self._master_addr
    env_vars["MASTER_PORT"] = self._master_port
```

#### 2. Smart Data Distribution

**Without RayWorkerGroup** - manual and error-prone:

```python
# Manual loop for distributing different shards
data_shards = [shard0, shard1, ..., shard7]
results = []
for i, worker in enumerate(workers):
    result = worker.process.remote(data_shards[i])
    results.append(result)
results = ray.get(results)
```

**With RayWorkerGroup** - automatic detection:

```python
# Distribute different shards (automatic detection!)
data_shards = [shard0, shard1, ..., shard7]  # Length matches worker count
results = worker_group.execute_all_sync("process", data_shards)

# Broadcast same data (automatic detection!)
results = worker_group.execute_all_sync("process", same_data)
```

The magic happens through automatic inspection:

```python
length = len(self._workers)
if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
    if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
        # Split args and kwargs into shards
        result = []
        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append(self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
        return result

return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]
```

#### 3. Placement Group Management

**Without RayWorkerGroup** - verbose manual setup:

```python
# Manually create placement groups for 10 nodes with 8 workers each
from ray.util.placement_group import placement_group

pg1 = placement_group([{"GPU": 1, "CPU": 10}] * 8, strategy="STRICT_PACK")
ray.get(pg1.ready())

workers = [
    ActorWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg1,
            placement_group_bundle_index=i
        ),
        num_gpus=0.125  # 8 workers sharing GPUs
    ).remote()
    for i in range(8)
]
# Repeat for each node... Very tedious!
```

**With RayWorkerGroup** - declarative specification:

```python
# All placement groups created and workers assigned automatically!
resource_pool = RayResourcePool(
    process_on_nodes=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],  # 10 nodes, 8 workers each
    use_gpu=True,
    max_colocate_count=8  # 8 workers share 1 GPU
)

worker_group = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=ray_cls
)
```

#### 4. Method Binding with Dispatch/Collect Patterns

The `RayWorkerGroup` automatically binds worker class methods and handles common distributed patterns. When you call a method on the worker group, it automatically:

1. **Dispatches** data appropriately (shard, broadcast, or pass-through)
2. **Executes** the method on all workers
3. **Collects** and merges results

This is powered by the `@register` decorator system.

### The @register Decorator: Declarative Distributed Execution

The `@register` decorator is a declarative system for distributed method execution. It allows you to specify **how** a method should be distributed and executed across workers without writing boilerplate code.

#### How It Works

**1. Metadata Attachment**

The decorator attaches metadata to methods:

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)
        
        # Attach metadata via magic attribute
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper
    
    return decorator
```

**2. Method Binding Process**

When `RayWorkerGroup` is initialized, it scans and binds decorated methods:

```python
def _bind_worker_method(self, user_defined_cls, func_generator):
    for method_name in dir(user_defined_cls):
        method = getattr(user_defined_cls, method_name)
        
        if hasattr(method, MAGIC_ATTR):
            # Extract configuration
            attribute = getattr(method, MAGIC_ATTR)
            dispatch_mode = attribute["dispatch_mode"]
            execute_mode = attribute["execute_mode"]
            blocking = attribute["blocking"]
            
            # Get dispatch and collect functions
            dispatch_fn = get_predefined_dispatch_fn(dispatch_mode)["dispatch_fn"]
            collect_fn = get_predefined_dispatch_fn(dispatch_mode)["collect_fn"]
            
            # Get execute function
            execute_fn = getattr(self, wg_execute_fn_name)
            
            # Bind method to WorkerGroup
            func = func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking)
            setattr(self, method_name, func)
```

**3. Execution Flow**

When you call a bound method, it follows this pipeline:

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            # Step 1: Dispatch - transform/distribute input data
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            
            # Step 2: Execute - run method on workers
            output = execute_fn(method_name, *args, **kwargs)
            
            # Step 3: Block (optional) - wait for results
            if blocking:
                output = ray.get(output)
            
            # Step 4: Collect - gather and merge results
            output = collect_fn(self, output)
            
            return output
    
    return Functor()
```

#### Common Dispatch Modes

**Dispatch.ONE_TO_ALL** - Broadcast same arguments to all workers:

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    # Load model weights - same operation on all workers
    pass

# Implementation
def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs
```

**Dispatch.DP_COMPUTE_PROTO** - Shard data across workers (data parallel):

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    # Each worker gets a different shard of data
    pass

# Automatically shards DataProto and concatenates results
```

**Dispatch.ALL_TO_ALL** - Pass-through (manual data distribution):

```python
@register(dispatch_mode=Dispatch.ALL_TO_ALL)
def custom_method(self, already_sharded_data):
    # Data already distributed manually
    pass
```

#### Why This Design?

The `@register` decorator system provides:

1. **Declarative**: Specify "what" (DP_COMPUTE_PROTO), not "how" to shard
2. **Consistent**: All methods follow the same pattern
3. **Type-safe**: Enforced dispatch/collect pairing
4. **Extensible**: Can register custom dispatch modes
5. **Clean code**: No boilerplate in worker methods

Without this system, every method would need manual sharding, dispatch, collect, and merge logic. The decorator makes distributed execution transparent and automatic.

---

With these building blocks understood, let's explore the core components that use them.

## Core Components

### ActorRolloutRefWorker

`ActorRolloutRefWorker` is a versatile hybrid engine that can serve three roles:

- **Actor**: The policy being learned (runs forward, backward, and parameter updates)
- **Rollout**: Pure inference engine (forward only, generates training sequences)
- **Reference**: Frozen copy of the initial policy (for KL divergence penalty computation)

This flexibility is crucial for efficient resource utilization in distributed RL training, where the same model infrastructure can serve different purposes.

#### Device Mesh Configuration

The initialization creates device meshes for different parallelism strategies:

**1. FSDP Device Mesh** - For distributed training:

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # Pure FSDP (all processes participate in sharding)
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        # Hybrid: DDP (across nodes) + FSDP (within nodes)
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh
```

**2. Ulysses Sequence Parallel Device Mesh** - For KV cache optimization:

```python
# Ulysses sequence parallelism distributes the sequence dimension across GPUs
# This reduces KV cache memory, which grows linearly with sequence length
self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
dp = world_size // self.ulysses_sequence_parallel_size

if self.ulysses_sequence_parallel_size > 1:
    self.ulysses_device_mesh = init_device_mesh(
        device_name, 
        mesh_shape=(dp, self.ulysses_sequence_parallel_size), 
        mesh_dim_names=["dp", "sp"]
    )
    self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
```

The benefit of Ulysses sequence parallelism is KV cache memory reduction. Since KV cache grows linearly with sequence length, distributing it across multiple GPUs enables training with longer sequences.

#### Model and Optimizer Initialization

The `_build_model_optimizer` method initializes the model using HuggingFace APIs and sets up the optimizer. It's called within `init_model`:

```python
# Model initialization with Flash Attention 2
actor_model_config = AutoConfig.from_pretrained(
    local_path, 
    trust_remote_code=trust_remote_code, 
    attn_implementation="flash_attention_2"
)

actor_module = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch_dtype,
    config=actor_model_config,
    trust_remote_code=trust_remote_code,
)
```

#### FSDP2 Configuration

After model initialization, FSDP2 is applied for distributed training. The sharding strategy depends on the device mesh dimensionality:

```python
def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    
    if device_mesh.ndim == 1:
        # Pure FSDP: shard across all processes
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        # Hybrid: DDP across first dimension, FSDP within second dimension
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
```

For FSDP2, mixed precision and CPU offload policies are configured:

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=param_dtype, 
    reduce_dtype=reduce_dtype, 
    cast_forward_inputs=True
)

# Critical: CPU offload handling
# - Reference model: Always use CPU offload to save memory
# - Actor model: NEVER use CPU offload when using gradient accumulation
#   Why? FSDP's CPU->GPU copy creates clean parameter copies, losing grad info!
cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

fsdp_kwargs = {
    "mesh": fsdp_mesh,
    "mp_policy": mp_policy,
    "offload_policy": cpu_offload,
    "reshard_after_forward": fsdp_config.reshard_after_forward,
}

# Apply FSDP2 and load state dict
full_state = actor_module.state_dict()
apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
```

**Optimizer**: Standard AdamW is used:

```python
actor_optimizer = optim.AdamW(
    actor_module_fsdp.parameters(),
    lr=optim_config.lr,
    betas=optim_config.get("betas", (0.9, 0.999)),
    weight_decay=optim_config.get("weight_decay", 1e-2),
)
```

#### Rollout Setup with vLLM

The `_build_rollout` method sets up the inference engine. The core components are `vLLMRollout` and `FSDPVLLMShardingManager`:

```python
# Initialize vLLM rollout engine (sync or async)
vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
rollout = vllm_rollout_cls(
    model_path=local_path,
    config=self.config.rollout,
    tokenizer=self.tokenizer,
    model_hf_config=self.actor_model_config,
    device_mesh=rollout_device_mesh,
    trust_remote_code=trust_remote_code,
    **lora_kwargs,  # Optional LoRA configuration
)

# Create sharding manager to sync FSDP training params with vLLM inference params
rollout_sharding_manager = FSDPVLLMShardingManager(
    module=self.actor_module_fsdp,
    inference_engine=rollout.inference_engine,
    model_config=self.actor_model_config,
    rollout_config=self.config.rollout,
    full_params=full_params,
    device_mesh=rollout_device_mesh,
    offload_param=self._is_offload_param,
    load_format=self.config.rollout.load_format,
    layered_summon=self.config.rollout.get("layered_summon", False),
)
```

**Important**: The rollout device mesh differs from the actor training device mesh:

- **Actor training**: All processes participate in FSDP parameter sharding
- **Rollout inference**: Uses tensor parallelism (TP) for fast inference, with data parallelism (DP) across TP groups

This design optimizes for different objectives: memory efficiency during training vs. throughput during inference.

#### Actor Initialization

If the worker serves as an actor, it creates a `DataParallelPPOActor` as the actual trainer:

```python
if self._is_actor:
    self.actor = DataParallelPPOActor(
        config=self.config.actor, 
        actor_module=self.actor_module_fsdp, 
        actor_optimizer=self.actor_optimizer
    )
```

### DataParallelPPOActor

`DataParallelPPOActor` serves as the actual trainer for the actor model. It can function as either an actor or a reference policy, depending on whether an optimizer is provided during initialization.

#### Forward Pass: Computing Log Probabilities

The `_forward_micro_batch` method computes log probabilities for generated responses:

```python
# Shift input_ids by 1 (standard autoregressive LM training)
# inplace_backward is a memory optimization - logits aren't needed after backward pass
log_probs = logprobs_from_logits(
    logits=logits_rmpad,
    labels=input_ids_rmpad_rolled,
    inplace_backward=inplace_backward,
)

# Extract log probs for response tokens only
log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
```

**Flash Attention Optimization**: When `use_remove_padding=True`, the attention mask is set to `None` and Flash Attention uses `cu_seqlens` for efficient computation. This is enabled via monkey patching:

```python
apply_monkey_patch(
    model=actor_module,
    use_remove_padding=use_remove_padding,
    ulysses_sp_size=self.ulysses_sequence_parallel_size,
    use_fused_kernels=use_fused_kernels,
    fused_kernels_backend=fused_kernels_backend,
)
```

#### Dynamic Batching for Load Balancing

The `compute_log_prob` method uses dynamic batching to balance computational workload. The core logic is in `verl.utils.seqlen_balancing.rearrange_micro_batches`:

**Key features**:

1. **Token-based splitting**: Split batch into micro-batches by total token count (not just batch size)
2. **Karmarkar-Karp Algorithm**: Balance sequence lengths across micro-batches
3. **Workload approximation**: Sort by sum of squared sequence lengths (approximates attention $O(n^2)$ complexity)
4. **DP synchronization**: Ensure all DP ranks use the same number of micro-batches

```python
# Calculate effective sequence lengths from attention mask (due to padding)
seq_len_effective = batch["attention_mask"].sum(dim=1)
total_seqlen = seq_len_effective.sum().item()

# Determine number of micro-batches based on max_token_len
num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))

# Synchronize across DP ranks to prevent out-of-sync
if dist.is_initialized() and same_micro_num_in_dp:
    num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
    dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)

# Use Karmarkar-Karp Algorithm for balanced partitioning
micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

# Sort by computational workload (sum of squared sequence lengths approximates O(n²) attention)
if use_dynamic_bsz_balance:
    micro_bsz_idx.sort(
        key=lambda partition: sum(seq_len_effective[idx] ** 2 for idx in partition),
        reverse=True,
    )

# Create micro-batches by reassembling samples
for partition in micro_bsz_idx:
    curr_micro_batch = torch.cat([batch[idx : idx + 1] for idx in partition])
    micro_batches.append(curr_micro_batch)
```

After batching, `_forward_micro_batch` performs the actual computation.

#### Policy Update: PPO Training Loop

The `update_policy` method implements the core PPO algorithm training logic:

```python
# Split batch into mini-batches for multiple PPO epochs
# See PPO paper: https://arxiv.org/abs/1707.06347
mini_batches = data.split(self.config.ppo_mini_batch_size)

for mini_batch in mini_batches:
    # Further split mini-batch into micro-batches for gradient accumulation
    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
    
    for micro_batch in micro_batches:
        # 1. Compute policy loss (PPO clipped objective)
        pg_loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss(...)
        
        # 2. Compute KL penalty from reference policy
        kl_penalty = compute_kl_penalty(...)
        
        # 3. Total loss and backward
        loss = pg_loss + kl_penalty
        loss.backward()
    
    # 4. Optimizer step after accumulating gradients
    self._optimizer_step()
```

#### PPO Loss Computation

The most challenging aspect of RL is understanding the loss formulations. For PPO, the implementation is in `verl.trainer.ppo.core_algos.compute_policy_loss`:

**Mathematical Formulation**:

The PPO loss with dual clipping is:

$$
L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]
$$

where:

- $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ is the probability ratio
- $A_t$ is the advantage estimate
- $\epsilon$ is the clip range (typically 0.2)

For dual-clip PPO (to maintain exploration), an additional constraint is added:

$$
L^{DUAL}(\theta) = \mathbb{E}_t[\max(L^{CLIP}(\theta), c \cdot A_t)] \text{ when } A_t < 0
$$

**Implementation**:

```python
# Compute probability ratio
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # Stability
ratio = torch.exp(negative_approx_kl)

# Standard PPO clipping
pg_losses1 = -advantages * ratio  # Unclipped objective
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # Clipped
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

# Dual-clip PPO: Prevent overly small ratio for negative advantages
# This maintains exploration by preventing the policy from becoming too deterministic
pg_losses3 = -advantages * clip_ratio_c  # clip_ratio_c typically = 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

# Apply dual-clip only for negative advantages
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

**Key insights**:

- Standard PPO clipping prevents large policy updates
- Dual-clip PPO (for negative advantages) prevents policy from becoming too deterministic
- The `response_mask` ensures only response tokens contribute to the loss (not prompt tokens)

### RayPPOTrainer

`RayPPOTrainer` orchestrates the entire PPO training process. It manages resource allocation, coordinates different worker groups (actor, critic, reward), and implements the main training loop.

#### Resource Management

`RayResourcePool` manages worker scheduling and allocation across GPU nodes using Ray Placement Groups. This ensures:

- Different actors get appropriate resources
- Co-location of compatible roles for efficiency (multiple roles can share the same resource pool)

#### Main Training Loop

The `fit` method implements the core training loop:

```python
self._load_checkpoint()

for epoch in range(num_epochs):
    for batch in self.train_dataloader:
        # 1. Generate sequences using current policy
        batch = self.actor_rollout_wg.generate_sequence(batch)
        
        # 2. Compute rewards
        reward_tensor, reward_extra_infos = compute_reward(batch, self.reward_fn)
        
        # 3. Compute log probabilities from old and reference policies
        batch = self.actor_rollout_wg.compute_log_prob(batch)  # old policy log probs
        batch = self.actor_rollout_wg.compute_ref_log_prob(batch)  # ref policy log probs
        
        # 4. Compute advantages using GAE or other estimators
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
        )
        
        # 5. Update critic (if using value function)
        if self.use_critic:
            critic_output = self.critic_wg.update_critic(batch)
        
        # 6. Update actor (after critic warmup)
        if self.global_steps >= self.config.trainer.critic_warmup:
            actor_output = self.actor_rollout_wg.update_actor(batch)
```

The trainer elegantly separates concerns through the `RayWorkerGroup` abstraction, making the training loop clean and maintainable.

### vLLMRollout

`vLLMRollout` integrates vLLM as the inference engine for efficient sequence generation during rollouts.

#### Initialization

For tensor parallelism (TP), vLLM setup varies by backend:

**Megatron backend**: Reuses existing process groups

```python
from vllm.distributed import parallel_state as vllm_ps

if kwargs.get("train_tp") is not None:
    # Reuse the process group already created by Megatron
    vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
```

**FSDP backend**: Delegates to vLLM's `LLM` class for process group initialization

#### Sequence Generation

The `generate_sequences` method is the main entry point for rollout generation:

```python
outputs = self.inference_engine.generate(
    prompts=vllm_inputs,  # Already converted to token IDs
    sampling_params=self.sampling_params,
    lora_request=lora_requests,
    use_tqdm=False,
)
```

**Important padding convention**:

- **Prompts**: Left-padded (for batch inference efficiency)
- **Responses**: Right-padded (standard autoregressive generation)

### FSDPVLLMShardingManager

The sharding manager is responsible for synchronizing FSDP training parameters with vLLM inference parameters. This is crucial in RL training because:

1. **FSDP parameters are distributed** across all ranks (each rank only has a shard)
2. **vLLM needs full or TP-sharded parameters** for inference
3. **Synchronization timing** affects both correctness and efficiency

#### Parameter Collection

The manager collects parameters from FSDP, handling both full model and LoRA cases:

```python
# Extract the actual model from FSDP wrapper
peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)

# Check if using LoRA (Parameter-Efficient Fine-Tuning)
if hasattr(peft_model, "peft_config"):
    peft_config = peft_model.peft_config.get("default", None)
    params = __collect_lora_params()  # Only collect LoRA adapter weights
else:
    params = self.module.state_dict()  # Collect full model weights

# Convert weight keys to match vLLM's expected format
params = convert_weight_keys(params, peft_model)
```

#### Parameter Update

The `update_params` method loads parameters into vLLM's inference engine. The key challenge is **gathering sharded parameters**:

```python
model = self.model_runner.model

# DTensor.full_tensor() gathers sharded parameters from all ranks
loaded_params = model.load_weights(
    (
        (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
        for name, param in updated_params.items()
    )
)
```

**Why DTensor?** In FSDP2, parameters are represented as `DTensor` (distributed tensor). Calling `.full_tensor()` performs an all-gather to reconstruct the full parameter, which vLLM needs for inference.

**Performance consideration**: The all-gather is expensive but necessary. VERL optimizes this with:

- **Layered summoning**: Gather parameters layer-by-layer to reduce peak memory
- **Selective updates**: Only update changed parameters when possible

### OneRecActorRolloutRefWorker

This is the key part how OpenOneRec implement RL for their post training. This could be used as a reference how to adopt RL for recommendation task.

OpenOneRec's customized worker extends `ActorRolloutRefWorker` to integrate the two-stage generation approach. The main change is overriding `_build_rollout`:

```python
def _build_rollout(self):
    # Override to use OneRecvLLMRollout instead of standard vLLMRollout
    rollout = OneRecvLLMRollout(
        model_path=local_path,
        config=self.config.rollout,
        tokenizer=self.tokenizer,
        model_hf_config=self.actor_model_config,
        device_mesh=rollout_device_mesh,
        trust_remote_code=trust_remote_code,
    )
    return rollout
```

This clean extension demonstrates VERL's extensibility - you can customize inference behavior without modifying the core training infrastructure.

### OneRecvLLMRollout: Two-Stage Generation

`OneRecvLLMRollout` implements OpenOneRec's two-stage generation approach:

1. **Stage 1**: Generate Chain-of-Thought (CoT) reasoning
2. **Stage 2**: Generate item ID sequences using beam search

This approach allows the model to first reason about the recommendation task before producing the actual item IDs, potentially improving recommendation quality.

#### Two-Stage Generation Flow

```bash
┌──────────────────────┐                    ┌─────────────┐
│ OneRecvLLMRollout    │                    │ vLLM Engine │
└──────────┬───────────┘                    └──────┬──────┘
           │                                       │
           │ ═══════ Stage 1: CoT Generation  ═════│
           │                                       │
      1.   │─── generate(prompt, stop="</think>") ─>
           │                                       │
      2.   │<──────── CoT reasoning tokens ────────│
           │                                       │
           │                                       │
           │ ═══════ Prepare Stage 2 Prompt  ══════│
           │                                       │
      3.   │─ prompt + CoT + "\n<|sid_begin|>"     │
           │                                       │
           │                                       │
           │ ═══ Stage 2: Item ID Generation  ═════│
           │                                       │
      4.   │─── beam_search(stage2_prompt, N)  ───>│
           │                                       │
      5.   │<──── N candidate item sequences ──────│
           │                                       │
      6.   │─ Select and format output             │
           │                                       │
           ▼                                       ▼
```

#### Stage 1: CoT Sampling

The first stage generates reasoning with a stop token:

```python
# Configure sampling for CoT generation
stage1_max_tokens = kwargs.get("stage1_max_tokens", getattr(self.config, "stage1_max_tokens", 1024))

cot_sampling_params = SamplingParams(
    n=1,  # Generate 1 CoT per prompt
    temperature=kwargs.get("temperature", 1.0),
    top_p=kwargs.get("top_p", 1.0),
    top_k=kwargs.get("top_k", -1),
    max_tokens=stage1_max_tokens,
    stop=["</think>"],  # Stop when reaching end of reasoning
    include_stop_str_in_output=True,  # Keep the stop token
)

# Generate CoT reasoning
cot_outputs = self.inference_engine.generate(
    prompts=vllm_inputs,
    sampling_params=cot_sampling_params,
    lora_request=lora_requests,
    use_tqdm=False,
)
```

**Key features**:

- **Stop token**: `</think>` marks the end of reasoning
- **Configurable length**: Stage 1 can have different max_tokens than Stage 2
- **Sampling**: Uses temperature/top_p for diverse reasoning

#### Preparing Stage 2 Input

After CoT generation, construct the Stage 2 prompt:

```python
tokenizer = self.inference_engine.get_tokenizer()
# Prefix marks the transition from reasoning to item ID generation
prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
vocab_size = len(tokenizer)

for i, output in enumerate(cot_outputs):
    cot_token_ids = list(output.outputs[0].token_ids)
    
    # Filter out-of-vocabulary (OOV) tokens
    # This can happen if vLLM generates special internal tokens
    cot_token_ids_filtered = [tid for tid in cot_token_ids if tid < vocab_size]
    
    cot_responses.append(cot_token_ids_filtered)
    
    # Construct Stage 2 prompt: [Original Prompt] + [CoT] + [Transition Prefix]
    original_prompt_ids = vllm_inputs[i]["prompt_token_ids"]
    new_prompt_ids = original_prompt_ids + cot_token_ids_filtered + prefix_ids
    
    stage2_input = {"prompt_token_ids": new_prompt_ids}
    # Preserve multi-modal data if present (e.g., images)
    if "multi_modal_data" in vllm_inputs[i]:
        stage2_input["multi_modal_data"] = vllm_inputs[i]["multi_modal_data"]
    
    stage2_inputs.append(stage2_input)
```

**Important steps**:

1. **OOV filtering**: Remove invalid token IDs that may be generated
2. **Prompt composition**: Concatenate original prompt + CoT + transition marker
3. **Multi-modal preservation**: Keep any images or other modalities

#### Stage 2: Beam Search for Item IDs

The second stage uses beam search to generate high-quality item sequences:

```python
beam_params = BeamSearchParams(
    beam_width=beam_width,
    max_tokens=max_tokens_item,
)

# Call beam search (aligned with standard implementation)
item_outputs = self.inference_engine.beam_search(
    prompts=stage2_inputs,
    params=beam_params,
)

# a relative complex post processing logic for the beam search result
if return_all_beams:
    # Return all beams, expand output
    # Output will be exactly batch_size * n_beams_to_return (pad if needed)
    expanded_idx = []
    beam_indices = []  # Track which beam index within each prompt

    for i, output in enumerate(item_outputs):
        # Prompt length including CoT + Prefix
        stage2_prompt_len = len(stage2_inputs[i]["prompt_token_ids"])
        original_prompt_len = len(vllm_inputs[i]["prompt_token_ids"])

        # Get top n beams for this prompt, pad if not enough
        num_seqs = len(output.sequences)
        for seq_idx in range(n_beams_to_return):
            if seq_idx < num_seqs:
                # this naming is not good, it is just get the beam_id's response
                best_seq = output.sequences[seq_idx]
                full_seq = best_seq.tokens
                # Response = full_seq - original_prompt (not stage2_prompt!)
                response_ids = full_seq[original_prompt_len:]
            else:
                # Pad with first beam's result if not enough beams
                best_seq = output.sequences[0]
                full_seq = best_seq.tokens
                response_ids = full_seq[original_prompt_len:]
            response.append(response_ids)
            expanded_idx.append(i)
            beam_indices.append(seq_idx)

    # Expand idx, attention_mask, position_ids to match expanded output
    idx = idx[expanded_idx]  # (batch_size * n, prompt_length)
    attention_mask = attention_mask[expanded_idx]
    position_ids = position_ids[expanded_idx]

    # Expand non_tensor_batch to match expanded output
    expanded_non_tensor_batch = {}
    for key, val in non_tensor_batch.items():
        if isinstance(val, np.ndarray):
            expanded_non_tensor_batch[key] = val[expanded_idx]
        elif isinstance(val, list):
            expanded_non_tensor_batch[key] = [val[i] for i in expanded_idx]
        else:
            expanded_non_tensor_batch[key] = val
    non_tensor_batch = expanded_non_tensor_batch

    # Store beam indices for reference
    non_tensor_batch["_beam_indices"] = np.array(beam_indices, dtype=np.int64)

    batch_size = len(response)  # Update batch_size

    print(f"[TwoStage] Expanded output: original_bs={len(item_outputs)}, expanded_bs={batch_size}, n_beams={n_beams_to_return}")
    
    ...
    
    seq = torch.cat([idx, response], dim=-1)
```

**Key considerations**:

- **Response extraction**: Use original prompt length, not stage2 prompt (which includes CoT)
- **Padding**: If fewer beams than requested, replicate the best beam
- **Metadata**: Track beam indices for reward computation or reranking

This two-stage approach enables:

1. **Better interpretability**: CoT reasoning explains the recommendation
2. **Higher quality**: Beam search explores multiple item sequence candidates
3. **Flexibility**: Can train rewards on both reasoning quality and recommendation accuracy

### onerec_recipe.py

This module contains domain-specific components for the recommendation task:

- **Dataloader**: Loads user interaction histories, candidate items, and formats prompts
- **Reward computation**: Calculates rewards based on relevance, diversity, and other recommendation metrics

These components are task-specific and can be customized for different recommendation scenarios.

---

## Summary

VERL provides a powerful and flexible infrastructure for distributed RL training through clean abstractions:

1. **RayWorkerGroup**: Simplifies Ray actor management with automatic coordination, smart data distribution, and placement group handling
2. **@register decorator**: Enables declarative distributed method execution with automatic dispatch/collect patterns
3. **Hybrid workers**: ActorRolloutRefWorker supports multiple roles (actor/rollout/ref) for resource efficiency
4. **FSDP2 + vLLM integration**: Seamlessly transitions between distributed training and efficient inference
5. **PPO implementation**: Complete with dynamic batching, dual-clip loss, and advantage estimation
6. **RayPPOTrainer**: Orchestrates the entire training process across multiple worker groups

OpenOneRec demonstrates how to extend VERL for domain-specific needs with minimal code changes, showcasing the framework's extensibility. The two-stage generation approach (CoT + beam search) illustrates how advanced generation strategies can be integrated into the RL training loop.

## References

- [VERL GitHub Repository](https://github.com/volcengine/verl)
- [OpenOneRec GitHub Repository](https://github.com/pyemma/OpenOneRec)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Dual-Clip PPO Paper](https://arxiv.org/pdf/1912.09729)
