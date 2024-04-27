# In-Batch Contractive Learning with DistributedDataParallel 

## Motivation

研究表明，增加批量大小（Batch Size）可以为批内对比学习（In-Batch Contractive Learning）任务提供更多具有挑战性的负样本，从而提高模型的性能。但是简单的利用DataParallel/DistributedDataParallel的数据并行策略（将一个批量（Batch）分成多个小批量（Mini Batch）使他们独立的在各个设备（GPU）上做训练）在本质上并没有增加单个样本的负样本数量。

因此，在利用分布式框架实现大批量的对比学习的时候还要在代码中增加一些通信的策略。这个过程里涉及了许多的细节和潜在的坑，本文以Instruction-Para CL的任务为例，介绍DDP（DistributedDataParallel）分布式框架下的大批量（批量内采样负样本）对比学习（In-Batch Contractive Learning with Big Batch Size)的实现方法。

## Main Idea

代码基本遵循DDP分布式训练的实现方法和套路，唯一需要注意的就是要在模型的前向传播过程中，将分发出去的数据（样本，正样，负样）的表征都收集（Gather）到各个设备上。然后每个进程的模型在做前向传播的时候就可以用整个批量（N*BS,假设共有N个设备，每个设备的批量大小均为BS），计算余弦相似度，利用交叉熵函数算对比学习的损失。

## Model Forward

整个过程最需要注意的就是Gather的操作，主流的通过DDP扩大BS做对比学习的gather方法一共有四种，可以参考下面这篇知乎博客，里面还包括了这几种Gather操作的梯度问题。

> [【DDP】PyTorch多卡分布式训练 | all_gather | 大batch对比学习 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/615784842)

下面，我将以上述文章的内容为主线，介绍在 DDP 框架下实现大批量对比学习的前向传要点。

### Single GPU

首先给出单GPU模式下模型的前向传播方法的实现：

```python
def forward(self, input_ids, token_type_ids, attention_mask, instruction_input_ids, instruction_token_type_ids,instruction_attention_mask):
    # instruction 前batch_size个是正样本，后batch_size个是困难负例
    instruction_output = self.encoder(input_ids=instruction_input_ids,
                                      attention_mask=instruction_attention_mask,
                                      token_type_ids=instruction_token_type_ids)

    para_output = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
    batch_size = input_ids.size(0)
    instruction_pooler = instruction_output.pooler_output  # [2*bs,768]
    para_pooler = para_output.pooler_output  # [bs,768]

    if self.add_batch:
        # 把同一个训练批次内的其余本身也作为负样, z2_shape: [3*bs,768]
        z2 = torch.cat((instruction_pooler, para_pooler), 0)
    cos_sim = self.sim(para_pooler.unsqueeze(1), z2.unsqueeze(0))
    # boardcast : [bs,1,768] , [1,3bs,768]  --> [bs,3bs]  if add_batch
    # boardcast : [bs,1,768] , [1,2bs,768]  --> [bs,2bs]  if not add_batch
    
    if self.hard_negative_weight != 0:
        # 设置困难负例的权重 给每个样本的困难负例的cos_sim + weight
        hard_tensor = cos_sim[:, batch_size:-batch_size]
        hard_tensor.diagonal().add_(self.hard_negative_weight)
    if self.add_batch:
        sub_tensor = cos_sim[:, -batch_size:]
        sub_tensor.diagonal().sub_(1e12)

    loss_fct = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(device)
    loss = loss_fct(cos_sim, labels)
    return loss
```



### Multi-GPU

DDP框架下的对比学习的前向传播会用到下面两个包。

```python
import torch.disributed as dist
import torch.distributed.nn
```

在单GPU前向传播的基础上，我们需要把各个设备上的样本、正例及负例的句子表征（Sentence Rresentation）（在这段代码分别对应了`para_pooler`、`instruction_pooler[: batch_size,:]`、以及`instruction_pooler[batch_size:,:]`）收集到一起，得到大batch的样本、正例及负例的句子表征。假设每个机子上我们分配了`per_device_batch_size`大小的批次，共有 `N` 个设备，那么Gather操作后的样本形状为 `N * per_device_batch_size`。分布式的前向传播代码如下所示。

```python
    def forward(self, input_ids, token_type_ids, attention_mask, instruction_input_ids, instruction_token_type_ids,
                instruction_attention_mask):
        
        per_device_batch_size = input_ids.size(0)  # 下面简写为pdbs
        # instruction的输入中， 前pdbs个是正例，后pdbs个是困难负例
        instruction_output = self.encoder(input_ids=instruction_input_ids,
                                          attention_mask=instruction_attention_mask,
                                          token_type_ids=instruction_token_type_ids)
        # instruction_output：[2*pdbs,768]
        para_output = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        # para_output：[pdbs,768]

        para_pooler = para_output.pooler_output  # [pdbs,768]
        positive_pooler = instruction_output.pooler_output[:per_device_batch_size]  
        # [pdbs,768]
        negative_pooler = instruction_output.pooler_output[per_device_batch_size:]  
        # [pdbs,768]

        # 如果使用了分布式训练并且模型处于训练模式
        if dist.is_initialized() and self.training:
            gathered_para_poolers = torch.distributed.nn.all_gather(para_pooler)
            gathered_positive_poolers = torch.distributed.nn.all_gather(positive_pooler)
            gathered_negative_poolers = torch.distributed.nn.all_gather(negative_pooler)
            # torch.distributed.nn.all_gather()的计算返回的结果为 Tutle[torch.tensor], 并且在反向传播的时候这个结果的梯度为求和后的
            para_pooler = torch.cat(gathered_para_poolers, 0)
            positive_pooler = torch.cat(gathered_positive_poolers, 0)
            negative_pooler = torch.cat(gathered_negative_poolers, 0)

        z2 = torch.cat((positive_pooler, negative_pooler), 0)
        # z2: [2*batch_size,768]
        if self.add_batch:
            # 把同一个训练批次内的其余本身也作为负样
            z2 = torch.cat((z2, para_pooler), 0)
            # z2: [3*bs,768]

        cos_sim = self.sim(para_pooler.unsqueeze(1), z2.unsqueeze(0))
        # boardcast : [bs,1,768] , [1,3bs,768]  --> [bs,3bs]  if add_batch
        # boardcast : [bs,1,768] , [1,2bs,768]  --> [bs,2bs]  if not add_batch

        gathered_scale = cos_sim.size(0)
        # 设置困难负例的权重 给每个样本的困难负例的cos_sim + weight
        if self.hard_negative_weight != 0:
            hard_tensor = cos_sim[:, gathered_scale:-gathered_scale]
            hard_tensor.diagonal().add_(self.hard_negative_weight)

        if self.add_batch:
            sub_tensor = cos_sim[:, -gathered_scale:]
            sub_tensor.diagonal().sub_(1e12)

        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(gathered_scale).long().to(int(os.environ["LOCAL_RANK"]))
        loss = loss_fct(cos_sim, labels)
        return loss
```

#### Code interpretation

DDP框架涉及的包主要是下面两个

```python
import torch.disributed as dist
import torch.distributed.nn
```

具体来说，DDP框架下的前向传播代码跟原代码的区别就在于DDP多了下面这段通信和处理。

考虑到简化参数传递的需要，我在传入指令时将正例指令和负例质量合并在一起。这样，instruction_output的前半部分表示正例，后半部分表示负例。由于使用了gather方法，因此在进行收集之前，需要将它们分开并逐个收集，否则gather后的指令结构可能会显得混乱。综合考虑，将它们在收集之前分开并逐个收集会比在收集完成后再逐个分开更为方便和清晰。

对于其他的代码，如果是单独对正例负例编码的话，就不需要前面的拆分过程。

```python
# 把正例负例从instruction里拆出来
positive_pooler = instruction_output.pooler_output[:per_device_batch_size] # [pdbs,768]
negative_pooler = instruction_output.pooler_output[per_device_batch_size:] # [pdbs,768]

# 如果使用了分布式训练并且模型处于训练模式
if dist.is_initialized() and self.training:
    gathered_para_poolers = torch.distributed.nn.all_gather(para_pooler)
    gathered_positive_poolers = torch.distributed.nn.all_gather(positive_pooler)
    gathered_negative_poolers = torch.distributed.nn.all_gather(negative_pooler)
    # torch.distributed.nn.all_gather()的计算返回的结果为 Tutle[torch.tensor], 
    # 且在反向传播的时候该方法的梯度为各设备梯度求和后的梯度（反向传播中做了一个all_reduce的操作，具体可以看下面对一系列gather方法的介绍）
    para_pooler = torch.cat(gathered_para_poolers, 0)
    positive_pooler = torch.cat(gathered_positive_poolers, 0)
    negative_pooler = torch.cat(gathered_negative_poolers, 0)

z2 = torch.cat((positive_pooler, negative_pooler), 0)
```

下面我将着重介绍一下通信部分的代码的执行和相关方法的介绍。

- `dist.is_initialized()`函数用于检查当前环境是否已经初始化了分布式环境。如果返回True，表示分布式环境已经初始化，可以继续进行后续的分布式操作；如果返回False，表示分布式环境尚未初始化，可能需要执行相应的初始化步骤。

- `self.training`用于表示当前模型是否处于训练模式。通过调用`model.train()`和`model.eval()`方法来切换模型的训练模式和推理模式，这两个方法会分别设置`self.training`属性为True和False。

- `torch.distributed.nn.all_gather()`函数的作用是将各个设备或节点上的数据按照指定的顺序收集到指定的目标位置中。这个函数通常用于收集每个设备或节点上的梯度或其他信息，以便在分布式环境中进行梯度聚合或其他操作。这个函数可以着重看一下上面这篇“【DDP】PyTorch多卡分布式训练 | all_gather | 大batch对比学习 ”对all_gather方法的介绍,包括他的前向传播和反向传播方法。总体来说，这个方法的的输出是一个元组，会把传进去的变量从各个设备上收集在输出的元组里，并且可以通过设备索引（local_rank）找到与索引对应的设备的这个变量的结果。以本文的代码为例，第i个GPU设备上的数据前向传播得到的`para_pooler`的值可以通过`gathered_para_poolers[0]`获取。

- 因为all_gahter的输出是一个元组，因此还需要将他们转换成张量，供后面的相似度和损失计算。这里采用torch.cat()的方法，当然torch.stack()在这里也是可以用的。

  

------

#### All_Gather

整体上代码需要进行的修改不多，但需要特别关注`all_gather`方法。总的来说，有以下几个需要注意的地方：

首先介绍一下PyTorch支持的两个all_gather方法：

- `torch.distributed.nn.all_gather(x)` 是 PyTorch 的分布式神经网络库 `torch.distributed.nn` 提供的函数之一。它通常用于在分布式神经网络的上下文中执行 AllGather 操作，例如在分布式训练过程中收集模型**<u>参数梯度</u>**。

  

- `dist.all_gather(x_list, x)`：这是另一个PyTorch官方的all_ather方法，不过这个方法并不**<u>直接</u>**适用于在对比学习前向传播的过程来扩大负样本数量。`dist.all_gather` 不会保留梯度，因为它是一个顶层的分布式函数，主要用于在分布式环境中收集数据。在模型训练中，通常不需要在分布式环境中收集梯度，因此这个函数**<u>不会保存梯度信息</u>**。因此，如果用了这个方法去做All Gather的话，要另外的处理梯度的信息。



利用上面这两个方法来实现对比学习前向传播过程中的All_Gather通信，基本参考本文提供的修改后代码就可以，这主要是基于`torch.distributed.nn.all_gather(x)`方法来实现的，同时也是比较推荐的一种。

如果使用`dist.all_gather(x_list, x)`方法的话来完成All_Gather通信的话，则需要做一定的包装。这里可以参考上面那篇博客介绍的四种All Gather实现，前三种分别是OPEN-CLIP、SimCLR和PyTorch-Lightning，这里的第四种其实就是`torch.distributed.nn.all_gather(x)`。

因为`torch.distributed.nn.all_gather(x)` 实际上就是通过 `dist.all_gather(x_list, x)` 构建一个 叫做`_ALL_GATHER`的`torch.autograd.Function ` 类来实现的。



------



下面我将简单地对上面四种方法进行介绍，如果需要更加具体地信息可以去看看上面的博客，或者源码。

##### OPEN CLIP 

```python
with torch.no_grad():
    all_x = [torch.zeros_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(all_x, x)
all_x[dist.get_rank()] = x
```

这是一个比较简单的方法，NLP领域内比较典型的对比学习方法**<u>SimCSE</u>**也是这么在前向传播的过程中实现All_Gather通信的。它的思路就是，先创建一个空的张量列表（`all_x`）用来容纳收集到的变量（`x`），但是由于`torch.distributed.all_gather(x_list,x)`不具有传递梯度的能力，因此all_x的每个元素都是没有梯度的。因此在最后的时候，还需要将有梯度的当前设备（获取设备索引可通过`dist.get_rank()`实现）的x与`all_x[dist.get_rank()]`做替换，从而把当前设备的有梯度的x传进`all_x`。但是也很显然的事，all_x其他位置的x都是没有梯度的。

用这套方法完成all_gather前向传播通信的完整过程可以参考[SimCSE/simcse/models.py at main · princeton-nlp/SimCSE (github.com)](https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L171)。

不过，还需要指出来的一点是，<u>**这样的实现方法虽然在逻辑上简洁易懂，但是可能存在缩放梯度的问题**</u>。梯度缩放的问题我会在后面的内容做一个简单的介绍。

对了，这里补充一个小细节，这里最上面的`with torch.no_grad()` 的作用仅仅只是做一个显示的声明，只是为了提示大家`torch.distributed.all_gather(all_x, x)` 这个过程是没有梯度的传递的。

##### SimCLR

```python
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
```

从SimCLR开始下面的三种方法都构建了一个专门的类，使我们可以显式的观察到他们的前向传播和传播的过程。这里再补充一个细小的知识点，对于这样的`Function`类 ，它的具体的运算过程要看它的`forward()`方法，`forward()`方法介绍了它的输入、输出和完整的计算过程；而他的梯度的运算或者说获取，就要看他的`backward()`方法，`backward()`展示了反向传播的时候会如何拿到这个`Function`运算的梯度。

很显然，SimCLR的前向传播和OPEN-CLIP几乎是一样的，就是利用`torch.distributed.all_gather()`方法，把变量收集到一起，但是少了一个传递梯度的过程。而这个梯度传递的过程，被SimCLR写到了`backward()`方法当中。SimCLR的backward()方法接收的grads是后一层运算返回的梯度，它通过将切片的方法获取当前设备的梯度，切片索引也就是当前设备的索引`dist.get_rank()`。

因此，从整体的逻辑的角度上看，SimCLR和OPEN-CILP的方法是等价的。如果说从OPEN-CILP上无法明显感觉到它的All Gather通信的梯度运算，就可以从SimCLR的反向传播入手。因此，OPEN-CILP和SimCLR都存在一个梯度缩放的问题。

SimCLR这套通信方法的使用方法：在前向传播过程中，对当前表示x执行`apply()`方法，其余不变，即可保留当前节点张量的梯度传播，最后将输出的元组转换成张量就可以继续往下运算了（如这里举例的求相似度运算）。

```python3
x = torch.cat(GatherLayer.apply(x), dim=0)
sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature
```

##### PyTorch-Lightning

```python
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
```

PyTorch-Lightening这套方法的`forward()`方法和SimCLR的`forward()`方法基本上是一致的，但是可以观察到PyTorch-Lightening的反向传播比SimCLR的反向传播多了一个all_reduce操作，正是这个求和的all_reduce操作帮助Pytorch-Lightening解决了梯度缩放的问题。这一点`torch.distributed.nn.all_gather()`方法在反向传播的过程与Pytorch-Lightening是保持一致的，都在反向传播的过程中，做了求和的All Redece，然后在对整个梯度进行切片，取出属于当前设备的那一份梯度。

##### torch.nn.distributed.all_gather

```python
def all_gather(tensor, group=group.WORLD):
    """
    Gathers tensors from the whole group in a list.
    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.
    Returns:
        tuple([Tensor]): Output of the collective.
    """
    return _AllGather.apply(group, tensor)
```

`torch.distributed.nn.all_gather()`方法应用了一个`_AllGather`的类，下面是`_AllGather`的源码。

```python
class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            rank = dist.get_rank()
            gx = torch.empty_like(grad_outputs[rank])
            _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)
```

以NCCL后端通信为例，`_AllGather`对梯度的求和的All Reduce是通过这一步`_Reduce_Scatter.apply(ReduceOp.SUM,...)`来完成的。简单来说，Reduce Scatter通信用于将多个进程中的张量按元素相加并Scatter到每个进程上,下面是_Reduce_Scater的源码。可以看出来，前向传播的过程中主要就是进行了一个`dist.reduce_scatter()`的操作来完成Reduce Scatter通信。

```python3
class _Reduce_Scatter(Function):
    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)
```

`torch.nn.distributed.all_gather`的使用可以参见Instruction-Para CL的forward代码(Multi-GPU那节)。简单来说就是下面这样：

```python
new_x = torch.distributed.nn.all_gather(x)
```



最后做一个前向传播All Gather的总结，就是要在模型的前向传播中设计All Gather的通信过程，使得模型在算损失前把分散在各个机子上的句子表征收集到一起，拿到一个大批量的负样然后计算对比学习损失。而在这个过程中，要格外关注梯度的传递问题，总的来说，`torch.nn.distributed.all_gather()`和PyTorch-Lightning的`SyncFunction`是可以使用的，而前两种（OPEN-CLIP和SimCLR）都存在一种梯度缩放的问题。如果要用前两种方法的话就需要在反向传播的时候给他们的梯度乘个设备数（也就是DDP框架下的World Size）。

## Scaling the Gradients

下面我将简单介绍一下All Gather通信可能会面临的梯度缩放问题。关于这一块的内容已有比较多的讨论，下面是比较经典的说明文章。

> [torch.distributed.nn.all_reduce incorrectly scales the gradient · Issue #58005 · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/issues/58005)
>
> 
>
> [Gradient backpropagation with torch.distributed.all_gather | by Jianfeng Wang | Medium](https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8)



第一篇github的issue针对这个问题给出了一些易于理解的案例，还有大家对于解决方法的讨论。最上面推荐的那篇知乎的文章最下面给出的案例也十分的能够说明问题。第二篇的文章在比较理论的角度说明了这个问题，并解释了为什么要乘以World Size。



下面我将以知乎文章下面最后的案例来说明为什么需要在反向传播的过程中，演示为什么要做一个求和的Reduce的步骤。

```python
import torch
import torch.distributed
import torch.distributed.nn

USE_NN_REDUCE = 1

def main_worker(gpu):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:12345", world_size=2, rank=gpu
    )
    torch.cuda.set_device(gpu)
    x = torch.ones(1).cuda(device=gpu).requires_grad_()
    xx = (gpu + 2) * x
    if USE_NN_REDUCE:
        xx = torch.distributed.nn.all_reduce(xx)
    else:
        torch.distributed.all_reduce(xx)
    y = (gpu + 2) * xx
    y.backward()
    print(f"gpu={gpu}, grad={x.grad}")

def local():
    x = torch.ones(1).cuda().requires_grad_()
    xx_all_reduce = 2 * x + 3 * x
    y_0 = 2 * xx_all_reduce
    y_1 = 3 * xx_all_reduce
    (y_0 + y_1).backward()
    print(f"truth: grad={x.grad}")

if __name__ == "__main__":
    local()
    torch.multiprocessing.spawn(main_worker, nprocs=2)
```



### DDP

假设有两个设备，从分布式的角度这段代码考虑了下面这个计算过程。
$$
x_{gpu} &=& 1 ， gpu = 0，1 \\
xx_{gpu}&=& (gpu + 2) * x \\
y_{gpu} &=& (gpu + 2) * xx \\
y &=& \sum y_{gpu}
$$
那么在第1个设备上时，，gpu=0，运算过程如下所示。
$$
xx_0&=& (0+2)*1 &=& 2~,~~此时~\frac{dxx_0}{dx} &=& 2 \\
y_0 &=& (0+2)*2 &=& 4~,~~此时~\frac{dy_0}{dxx_0} &=& 2 \\
$$
在第2个设备上，gpu=1，运算过程如下所示。
$$
xx_1 &=& (1+2)*1 &=& 3~,~~此时~\frac{dxx_1}{dx} &=& 3 \\
y_1 &=& (1+2)*2 &=& 6~,~~此时~\frac{dy_1}{dxx_1} &=& 3 \\
$$
最后有：
$$
y=y_0+y_1 = 4+6 = 10
$$
然后我们进一步考虑反向传播过程中梯度的计算，一共有下面两种情况。

如果进行了求和的Ruduce的话，而使用了`torch.distributed.nn.all_reduce`方法，在进行`y.backward()`的操作之前，`torch.distributed.nn.all_reduce(xx)` 会先把各个设备上xx对x的梯度进行求和，导致各个设备上的xx的梯度全都变成5 。这个过程可以被描述成下面的表达式。
$$
\frac{dxx_{0}}{dx} = \frac{dxx_{1}}{dx} = 2+3 = 5 \\
\frac{dy_{0}}{dx} = \frac{dy_{0}}{dxx} \times \frac{dxx_{0}}{dx} = 2\times5 =10 \\
\frac{dy_{1}}{dx} = \frac{dy_{1}}{dxx} \times \frac{dxx_{1}}{dx} = 3\times5 =15 \\
\frac{dy}{dx} = \frac{dy_0}{dx} + \frac{dy_1}{dx} = 10 + 15 = 25
$$
如果没有求和的Ruduce的话，而直接使用`dist.all_reduce`方法就会导致下面的计算过程和结果：
$$
\frac{dy_{0}}{dx} = \frac{dy_{0}}{dxx} \times \frac{dxx_{0}}{dx} = 2\times2 =4 \\
\frac{dy_{1}}{dx} = \frac{dy_{1}}{dxx} \times \frac{dxx_{1}}{dx} = 3\times3 =9 \\
\frac{dy}{dx} = \frac{dy_0}{dx} + \frac{dy_1}{dx} = 4 + 9 =13
$$
<u>**Tips: y.backward() 的过程中会使用Ring_allReduce,默认把各个设备上的y的梯度相加。**</u>

### Single GPU

如果不考虑分布式，演示代码的计算过程可以被简化下面的表达式，也就是`local()`执行的计算：
$$
\begin{align*}
y &= y_0 + y_1 \\
&= 2xx + 3xx \\
&= 5xx \\
&= 5\times(2x+3x) \\
&= 25x \\
\end{align*}
$$
因此，
$$
\frac{dy}{dx}=25
$$


通过上面的比对，在每一次的收集All Gather时都需要同时考虑反向传播时梯度的Sum的Reduce操作，否则会导致错误的梯度，对训练产生影响。



PS:对于为什么给梯度乘一个World Size也可以解决梯度缩放的问题，或者说为什么Sum的Reduce操作和乘上一个World Size等价 我也还不太明白，留待确认和补充.....              

## Training                      

用DDP框架训练，只要在原来的代码(非DDP)的基础上做一些修改就可以，当然用HuggingFace的Accelerate包完成这个过程会更加容易（增加的代码量更少）。下面，我将介绍将单GPU训练代码改为DDP分布式训练的几个步骤。

### logging

只让主进程打印和保存

### Batch Size

每个进程都开batchsize的批量 通信以后聚合在一起全局的批次就是world_size*per_device_batch_size

### seed





