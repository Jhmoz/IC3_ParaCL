# In-Batch Contractive Learning with DistributedDataParallel  : Train

## Motivation

研究表明，增加批量大小（Batch Size）可以为批内对比学习（In-Batch Contractive Learning）任务提供更多具有挑战性的负样本，从而提高模型的性能。但是简单的利用DataParallel/DistributedDataParallel的数据并行策略（将一个批量（Batch）分成多个小批量（Mini Batch）使他们独立的在各个设备（GPU）上做训练）在本质上并没有增加单个样本的负样本数量。

因此，在利用分布式框架实现大批量的对比学习的时候还要在代码中增加一些通信的策略。这个过程里涉及了许多的细节和潜在的坑，本文以Instruction-Para CL的任务为例，介绍DDP（DistributedDataParallel）分布式框架下的大批量（批量内采样负样本）对比学习（In-Batch Contractive Learning with Big Batch Size)的实现方法。

