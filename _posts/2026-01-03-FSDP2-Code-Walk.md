---
title: FSDP2 Under the Hood - A Deep Dive into PyTorch's Fully Sharded Data Parallel Implementation
date: 2026-01-03
author: pyemma
categories: [Distributed Training]
tags: [distributed-training, pytorch, fsdp, model-parallelism]
toc: true
math: true
---

Fully Sharded Data Parallel (FSDP) is PyTorch's approach to training large models that don't fit in a single GPU's memory. FSDP2 represents a significant redesign from FSDP1, with improved performance, better composability, and a cleaner architecture built on top of PyTorch's DTensor abstraction. In this post, I'll walk through the implementation details of FSDP2, exploring how it shards parameters, orchestrates communication, and overlaps computation with communication to achieve efficient distributed training. Please feel free to correct me if there are any misunderstanding.

FSDP2's core philosophy is similar to *parameter-server* architecture, where the parameters are allocated *somewhere else* instead of the current node to reduce the memory overhead. The parameter is fetched on-demand when they are required for compute, and freed when they are no longer used.

> PS: I used cursor to help me navigate the entire pytorch codebase and explain certain syntax and business logic that I'm not familiar with. I spend around 6 hours in total to have a full E2E understand on how FSDP2 internally works, which is impossible without AI. I'm pretty impressed on this journey and it would reshape how in the future I would work and study.

## High-Level Architecture

FSDP2 is built around several core components that work together to manage parameter sharding and communication:

### Core Components

**FSDPState** - The central state manager that tracks all FSDP-related metadata for a module. It maintains references to parameter groups, manages hook registration, and coordinates the overall FSDP execution flow.

**FSDPParamGroup** - Groups multiple parameters together for efficient collective communication. This implements the "bucketing" strategy where parameters are flattened into a single tensor before communication, reducing the number of collective operations needed.

**FSDPParam** - Wraps individual parameters and manages their sharding/unsharding lifecycle. Each `FSDPParam` encapsulates a `DTensor` that represents the sharded parameter, and handles the conversion between sharded and unsharded states.

**FSDPModule** - A wrapper that transforms regular PyTorch modules into FSDP-aware modules by dynamically creating new module classes with the "FSDP" prefix.

### Key Design Principles

1. **Hook-based Execution**: FSDP2 uses PyTorch's forward/backward hooks to intercept module execution. Pre-forward hooks handle parameter unsharding, while post-forward hooks handle resharding to free memory.

2. **DTensor Integration**: FSDP2 leverages PyTorch's DTensor abstraction for sharding semantics. DTensors provide a clean way to represent distributed tensors with sharding specifications, and integrate seamlessly with autograd. However, one key thing here is that all parameters' communication is bucketized together as `FSDPParamGroup` instead of using the individual API exposed via `DTensor`. This is a key optimization from FSDP2 because of the global view on the parameters.

3. **Stream-based Communication Overlap**: FSDP2 uses separate CUDA streams for communication operations, allowing computation and communication to overlap. The `all_gather_stream` handles collective communication, while `all_gather_copy_in_stream` prepares data for the next layer's communication.

## Initialization Flow

The entry point to FSDP2 is the `fully_shard()` function. Let's trace through what happens when you wrap a module with FSDP:

```python
  modules = (
    (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
  )
  state = fully_shard.state(modules[0])
  state.init(modules, device, mp_policy, auto_reshard_after_forward)
  ...
  if params:
      state._fsdp_param_group = FSDPParamGroup(
          params,
          modules,
          mesh_info,
          post_forward_mesh_info,
          device,
          shard_placement_fn,
          mp_policy,
          offload_policy,
      )
  for module in modules:
      cls = module.__class__
      new_cls = cls_to_fsdp_cls.get(cls)
      if not new_cls:
          dct = {"__deepcopy__": _unimplemented_deepcopy}
          new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
          cls_to_fsdp_cls[cls] = new_cls
          module.__class__ = new_cls
  ```

The initialization process involves:

1. **State Initialization**: Creates an `FSDPState` object and registers pre/post forward/backward hooks on the module.

2. **Parameter Discovery**: Finds all parameters that need to be sharded and groups them into an `FSDPParamGroup`.

3. **Module Transformation**: Dynamically creates a new module class that inherits from both `FSDPModule` and the original module class, then replaces the module's class. This allows FSDP to intercept module operations while preserving the original module's functionality.

4. **DeviceMesh and ProcessGroup Setup**: The `DeviceMesh` abstraction manages the process group for communication. The `FSDPParamGroup` retrieves its process group through the mesh info:

```python
      @property
      def _all_gather_process_group(self) -> dist.ProcessGroup:
          mesh_info = (
              cast(FSDPMeshInfo, self.post_forward_mesh_info)
              if self.is_sharded_post_forward
              else self.mesh_info
          )
          if not isinstance(mesh_info, FSDPMeshInfo):
              raise AssertionError(
                  f"Expected mesh_info to be FSDPMeshInfo, got {type(mesh_info)}"
              )
          return mesh_info.shard_process_group
```

## Parameter Sharding

`FSDPParam` is the key component within `FSDPParamGroup`. Each parameter is sharded via `FSDPParam._init_sharded_param()`. This is where DTensor integration happens:

  ```python
  # shard the param data based on the world size and the dim to shard
  chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
  sharded_param = chunks[shard_rank]
  # there is some additional padding logic here
  ...
  # finally create the DTensor here and register it to the module
  self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
  self.sharded_param.requires_grad_(param.requires_grad)
  self._setattr_on_modules(self.sharded_param)
  self.sharded_state = ShardedState.SHARDED
  ```

During chunking, only shard 0 supports uneven sizes. For other dimensions, the size must be divisible by the world size. Shard 0 is always padded, and this padded size is used to pad other shards:

```python
  padded_sharded_size = chunks[0].size()  # 0th always padded
  self.padded_sharded_param_size = padded_sharded_size
  # Pre-pad the sharded parameter to avoid padding before all-gather
  padded_sharded_param = param_data.new_zeros(padded_sharded_size)
```

When the input parameter is already a `DTensor`, FSDP2 handles FSDP + Tensor Parallel hybrid parallelism. The device mesh needs to be configured accordingly, e.g., `[4, 2], [ShardDim(0), ShardDim(1)]` for a 2D mesh. (I will have another blog on TP using `DTensor`)

Note that `self._sharded_param_data` stores the flattened tensor (used for all-gather operations), while `self.sharded_param` preserves the parameter with original shape.

## Forward Pass Execution

The forward pass in FSDP2 is orchestrated through hooks that manage parameter unsharding, communication, and resharding. These hooks are defined within `FSDPState` and registered by function `_register_group_forward_hooks` and `_register_pre_backward_hook`.

### Pre-Forward Hook

The `_pre_forward` hook is where the magic happens. It coordinates several critical operations:

```python
def _pre_forward(
    self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # When composing with module-hook-based activation checkpointing, the
    # pre-backward hook is responsible for the unshard
    if self._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    self._training_state = TrainingState.FORWARD
    args, kwargs = self._root_pre_forward(module, args, kwargs)
    if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(
                _cast_fp_tensor, self._mp_policy.param_dtype
            )
            args, kwargs = (
                _apply_to_tensors(cast_fn, args),
                _apply_to_tensors(cast_fn, kwargs),
            )
    if self._fsdp_param_group:
        args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
    for fsdp_state in self._states_to_forward_prefetch:
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
    return args, kwargs
```

The hook performs several key tasks:

1. **Stream Synchronization**: Before starting forward pass, FSDP2 synchronizes communication streams with the optimizer stream to ensure parameters have been updated:

```python
# Wait for optimizer before implicitly prefetched all-gathers
if (event := self._state_ctx.post_optim_event) is not None:
    self._comm_ctx.all_gather_copy_in_stream.wait_event(event)
    self._comm_ctx.all_gather_stream.wait_event(event)
    self._state_ctx.post_optim_event = None
          else:
    current_stream = self._device_handle.current_stream()
    self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
    self._comm_ctx.all_gather_stream.wait_stream(current_stream)
```

This ensures that all-gather operations read the updated parameters after the optimizer step, avoiding stale data.

2. **Mixed Precision Handling**: If mixed precision is enabled, the hook casts input tensors to the appropriate dtype.

3. **Parameter Unsharding**: The `FSDPParamGroup.pre_forward()` method handles unsharding (`self.unsard` and `self.wait_for_unshard`):

```python
      def pre_forward(
          self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
      ) -> tuple[tuple[Any, ...], dict[str, Any]]:
          if not compiled_autograd_enabled():
              logger.debug("%s", self._with_fqn("FSDP::pre_forward"))
          with record_function(self._with_fqn("FSDP::pre_forward")):
              self._training_state = TrainingState.FORWARD
              self.unshard(self.unshard_async_op)
              self.wait_for_unshard()
              args, kwargs = self._register_post_backward_hook(args, kwargs)
              return args, kwargs
```

4. **Prefetching**: The hook can prefetch parameters for the next layer to overlap communication with computation.

### All-Gather Communication

The `unshard()` method triggers the all-gather operation. The core communication logic is in `foreach_all_gather()`:

```python
  @torch.no_grad()
  def foreach_all_gather(
      fsdp_params: list[FSDPParam],
      group: dist.ProcessGroup,
      async_op: bool,
      all_gather_copy_in_stream: torch.Stream,
      all_gather_stream: torch.Stream,
      device: torch.device,
      all_gather_comm: AllGather,
  ) -> Optional[AllGatherResult]:
      world_size, rank = group.size(), group.rank()
      device_handle = _get_device_handle(device.type)
      # this is a context manager, and all kernel launch would be sent to `all_gather_copy_in_stream`
      with device_handle.stream(all_gather_copy_in_stream):
          # this function create a flatten version of parameters, and split for separation
          param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
          (
              param_all_gather_input_dtypes,
              param_all_gather_input_numels,
              dtype,
          ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
          if dtype == torch.uint8:
              all_gather_inputs = [
                  t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
              ]
          else:
              all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
          inp_split_sizes = [t.numel() for t in all_gather_inputs]
          all_gather_input_numel = sum(inp_split_sizes)
          # this is a interface that design for the task of communication primitives,
          # which defines how, where to allocate memory
          all_gather_output = all_gather_comm.allocate(
              (all_gather_input_numel * world_size,), dtype=dtype, device=device
          )
          all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
              all_gather_inputs,
              all_gather_output,
              inp_split_sizes,
              all_gather_input_numel,
              rank,
          )
          del param_all_gather_inputs
      all_gather_stream.wait_stream(all_gather_copy_in_stream)
      with device_handle.stream(all_gather_stream):
          # this would invoke dist.all_gather_into_tensor 
          all_gather_work = all_gather_comm(
              output_tensor=all_gather_output,
              input_tensor=all_gather_input,
              group=group,
              async_op=async_op,
          )
          all_gather_event = all_gather_stream.record_event()
          return AllGatherResult(
              all_gather_output,
              all_gather_event,
              all_gather_work,
              param_all_gather_input_dtypes,
              param_all_gather_input_numels,
              inp_split_sizes,
          )
```

The process involves:

1. **Flattening Parameters**: All parameters in the group are flattened into a single tensor for efficient communication.

2. **Copy-In**: The `all_gather_copy_in` operation copies each rank's local shard into the appropriate position of the all-gather output buffer. This happens on the `all_gather_copy_in_stream`.

3. **Communication**: The actual all-gather collective is executed on the `all_gather_stream`, which waits for the copy-in stream to complete.

4. **Copy-Out**: After communication completes, `wait_for_unshard()` calls `foreach_all_gather_copy_out()` to distribute the gathered data back to individual parameters:

```python
# get all necessary information from the all_gather_result
(
    all_gather_output,
    all_gather_event,
    all_gather_work,
    param_all_gather_input_dtypes,
    param_all_gather_input_numels,
    all_gather_input_split_sizes,
) = all_gather_result

for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
    param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
):
    # initialize the all gather outputs buffer in the FSDPParam
    fsdp_param.init_all_gather_outputs(
        all_gather_input_numels,
        all_gather_input_dtypes,
        world_size,
        device,
        force_recreate=force_recreate,
    )
    if not force_recreate:
        fsdp_param.alloc_all_gather_outputs()
    # param_all_gather_outputs point to the same reference as fsdp_param.all_gather_outputs
    param_all_gather_outputs = fsdp_param.all_gather_outputs
    split_with_sizes_out.extend(param_all_gather_outputs)

# change the flatten all gather output buffer into (world_size, all_params_numel)
all_gather_output = all_gather_output.view(world_size, -1)
# also change each parameters to (world_size, original_param_size)
out = [t.view(world_size, -1) for t in split_with_sizes_out]
# split from the all_params_numel to the actual params size, which is stored in all_gather_input_split_sizes
# and copy to the out (to make sure the shape is aligned)
torch.ops.fsdp.split_with_sizes_copy(
    all_gather_output, all_gather_input_split_sizes, dim=1, out=out
)
```

The `init_unsharded_param()` method then creates the unsharded parameter from the all-gather output:

```python
unsharded_tensor = self.all_gather_outputs[0]
unsharded_param = torch.as_strided(
    unsharded_tensor,
    self._orig_size,
    self._contiguous_orig_stride,
    storage_offset=0,
)
self._unsharded_param = nn.Parameter(
    unsharded_param, requires_grad=self.sharded_param.requires_grad
)
```

Note: The context manager `torch.autograd._unsafe_preserve_version_counter()` is used to tell autograd to ignore the inplace operation when copying data to the parameter location.

### Post-Forward Hook

After the forward pass completes, the `_post_forward` hook reshards parameters to free memory:

```python
      def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
          # When composing with module-hook-based activation checkpointing, the
          # post-backward hook is responsible for the reshard
          if self._training_state == TrainingState.PRE_BACKWARD:
              return output
          if self._fsdp_param_group:
              output = self._fsdp_param_group.post_forward(module, input, output)
          output = self._register_pre_backward_hook(output)
          self._training_state = TrainingState.IDLE
          if self._state_ctx.iter_forward_root is self:
              if all_gather_state := self._comm_ctx.all_gather_state:
                  # Free the last all-gather result if needed; refer to
                  # [Note: Overlapping all-gather copy-in and all-gather]
                  self._comm_ctx.all_gather_copy_in_stream.wait_event(
                      all_gather_state.event
                  )
                  self._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
                  self._comm_ctx.all_gather_state = None  # free the all-gather result
              self._state_ctx.iter_forward_root = None
          if self._mp_policy.output_dtype is not None:
              with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                  output = _apply_to_tensors(
                      functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                      output,
                  )
          return output
```

The `reshard()` method converts parameters back to sharded form:

```python
def reshard(self):
    if self._training_state == TrainingState.FORWARD:
        if not self._reshard_after_forward:
            return
        if self._use_post_forward_mesh:
            self._to_sharded_post_forward()
            self._reshard_after_forward_event = self.device_handle.Event()
            if self._reshard_after_forward_event is not None:
                self._reshard_after_forward_event.record()
            return
    self._to_sharded()
```

## Backward Pass Execution

The backward pass follows a similar pattern to the forward pass. The `_pre_backward` hook is registered on the output tensors and triggers when gradients start flowing backward. It performs parameter unsharding similar to the forward pass, but with one key difference: **implicit prefetching**.

In the forward pass, prefetching must be explicitly set up because we don't know which layers will be executed next. However, in the backward pass, the forward computation graph is already known, so FSDP2 can automatically determine which parameters need to be prefetched and schedule the all-gather operations accordingly.

The backward pass also handles gradient communication through all-reduce operations, which are orchestrated through the same communication infrastructure.

## Communication Primitives

Let's trace the communication stack from FSDP2 down to the hardware:

### High-Level Flow

The communication path is:

```text
FSDPParamGroup.unshard() 
  → foreach_all_gather() 
  → AllGather.__call__() 
  → dist.all_gather_into_tensor() 
  → torch.distributed.distributed_c10d.all_gather_into_tensor() 
  → ProcessGroup._allgather_base() 
  → ProcessGroup C++ binding 
  → ProcessGroupNCCL (or other backend)
```

The `AllGather` object (defaulting to `DefaultAllGather`) provides an abstraction over the communication primitive, allowing different memory allocation strategies (e.g., RDMA-based allocation for NCCL).

### PyTorch Distributed Stack

**Python Layer**: `torch.distributed.distributed_c10d` provides Python bindings for distributed operations.

**C++ ProcessGroup Abstraction**: The `ProcessGroup` class in `torch/csrc/distributed/c10d/ProcessGroup.hpp` provides a virtual interface for communication backends. When you call `dist.broadcast(tensor, src)`, it goes through PyBind11 bindings to the C++ `ProcessGroup::broadcast()` method:

```cpp
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("c10d::broadcast_", "")
              .typed<
                  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                      at::TensorList,
                      const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                      int64_t,
                      int64_t,
                      bool,
                      int64_t)>();
  ```

The operations are registered via macro functions in `Ops.cpp`:

```cpp
  #define IMPL_BROADCAST(DEV)                                               \
    std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>           \
        broadcast_##DEV(                                                    \
            at::TensorList tensors,                                         \
            const c10::intrusive_ptr<ProcessGroup>& process_group,          \
            int64_t root_rank,                                              \
            int64_t root_tensor,                                            \
            bool asyncOp,                                                   \
            int64_t timeout) {                                              \
      auto tensor_vec = tensors.vec();                                      \
      auto work = process_group->getBackend(c10::DeviceType::DEV)           \
                      ->broadcast(                                          \
                          tensor_vec,                                       \
                          BroadcastOptions{                                 \
                              root_rank,                                    \
                              root_tensor,                                  \
                              std::chrono::milliseconds(timeout),           \
                              asyncOp});                                    \
      return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>( \
          std::move(tensor_vec), work);                                     \
    }
  ```

**NCCL Backend**: `ProcessGroupNCCL` implements the actual GPU communication using NCCL. It uses a template function `collective()` as the foundation for all communication primitives:

```cpp
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle,
      bool nanCheck)
  ```

Here, `Fn`, `PreProcess`, and `PostProcess` are lambda functions. Different communication primitives provide different `Fn` implementations. For example, `allreduce_impl` passes `ncclAllReduce` as the `Fn`:

```cpp
[&](at::Tensor& input,
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream) {
  auto ncclDataType = getNcclDataType(input.scalar_type());
  auto ncclReduceOp =
      getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
  return ncclAllReduce(
      input.data_ptr(),
      output.data_ptr(),
      input.numel(),
      ncclDataType,
      ncclReduceOp,
      comm,
      stream.stream());
},
```

### Stream Management

FSDP2 uses separate CUDA streams for communication to enable overlap:

- **`all_gather_stream`**: Handles the actual collective communication operations for the current layer.
- **`all_gather_copy_in_stream`**: Prepares data for the next layer's communication (e.g., flattening buffers).

When `asyncOp` is enabled, NCCL uses a dedicated stream instead of the current compute stream:

```cpp
// in asyncOp=false [default] mode, we use currentStream as ncclStream
// otherwise, we use separate ncclStream and let it sync on currentStream
auto ncclStream = asyncOp ? ncclStreams_.at(key)
                          : at::cuda::getCurrentCUDAStream(device.index());
if (asyncOp) {
  // First let NCCL streams wait for input tensors allocation streams
  syncStream(device, ncclEvents_[key], ncclStream);
}
```

This allows computation and communication to overlap, with proper synchronization to ensure data dependencies are respected.

## Advanced Topics

### Prefetching Strategies

FSDP2 supports two types of prefetching:

**Explicit Prefetching (Forward Pass)**: Since the forward computation graph isn't known ahead of time, prefetching must be manually configured. The `_pre_forward` hook checks `self._states_to_forward_prefetch` (set via `FSDPModule.set_modules_to_forward_prefetch()`) and calls `FSDPParamGroup._prefetch_unshard()` for the next layer. Also the all-gather happens on different cuda stream, this makes it possible for CPU to schedule all-gather for next layer and compute for the current layer concurrently. This is a key to overlap compute and communication to hidden latency.

**Implicit Prefetching (Backward Pass)**: After the forward pass completes, the computation graph is known. FSDP2 can automatically determine which parameters need to be prefetched and schedule all-gather operations on a separate stream while the current layer's computation runs on the default stream, achieving overlap.

The effectiveness of implicit prefetching depends on whether the workload is CPU-bound or GPU-bound. For CPU-bound workloads, explicit prefetching setup may be necessary.

### Memory Management

**Parameter Bucketing**: `FSDPParamGroup` groups multiple parameters together and flattens them into a single tensor before communication. This reduces the number of collective operations and improves efficiency, similar to DDP's bucketing strategy.

## Summary

FSDP2 represents a significant evolution in PyTorch's distributed training capabilities. By building on DTensor and using a clean hook-based architecture, it provides:

- **Memory Efficiency**: Parameters are sharded and only unsharded when needed
- **Communication Efficiency**: Parameter bucketing and stream-based overlap minimize communication overhead
- **Composability**: Clean integration with other PyTorch features like mixed precision and activation checkpointing
- **Flexibility**: Support for hybrid parallelism (FSDP + TP) and various memory allocation strategies

> If you find this post helpful, feel free to scan the QR code below to support me and treat me to a cup of coffee
{: .prompt-tip }

![Thank You](/assets/qr%20code.png){: width="300" height="300" }

## References

1. [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
2. [PyTorch DTensor Documentation](https://pytorch.org/tutorials/prototype/dtensor_docs.html)
3. [PyTorch Distributed Communication (c10d)](https://pytorch.org/docs/stable/distributed.html)
4. [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
5. PyTorch Source Code:
   - `torch.distributed.fsdp._fully_shard`
   - `torch.distributed.fsdp._fully_shard._fsdp_state.py`
   - `torch.distributed.fsdp._fully_shard._fsdp_param_group.py`
   - `torch.distributed.fsdp._fsdp_collectives.py`
   - `torch/csrc/distributed/c10d/ProcessGroup.hpp`
