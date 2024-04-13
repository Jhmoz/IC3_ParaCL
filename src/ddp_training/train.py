from dataset_fn import *
from model_class import *
import argparse
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.optim as optim
import os
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--infor_file", type=str, default=r'../data/Tag_CLData.json', help='summary所在的json文件')
    parser.add_argument("--para_file", type=str, default=r'../data/instruction_para.json',
                        help='para所在的json文件')
    parser.add_argument("--pretrained", type=str, default=r'/home/yingying/models/chinese-roberta-wwm-ext/',
                        help='选取的基准模型')
    parser.add_argument('--long_semantic', action="store_true", help='是否展开长答案的学习')
    parser.add_argument('--short_semantic', action="store_true", help='是否展开短答案的学习')
    parser.add_argument("--output_dir", type=str, default=r'./output', help='模型保存位置')
    parser.add_argument("--batch_size", type=int, default=24, help='batch_size')
    parser.add_argument("--lr", type=float, default=1e-5, help='学习率大小')
    parser.add_argument("--epoch", type=int, default=3, help='epoch数')
    parser.add_argument("--add_batch", action="store_true", help='是否将template作为负样本')
    parser.add_argument("--logging_step", type=int, default=500, help='训练过程中打印loss的间隔步数')
    parser.add_argument("--hard_negative_weight", type=float, default=0, help='给每个困难负例增加权重，提高任务难度')
    parser.add_argument("--local_rank", type=int, default=-1, help='DDP分布式训练环境下每个进程需要自动获取的设备号')
    parser.add_argument("--logging_rank", type=int, default=0, help='DDP分布式训练环境下用于打印日志的全局设备号')
    args = parser.parse_args()
    return args


def print_rank_0(info, rank):
    if int(os.environ["RANK"]) == rank:
        logging = "Rank {}: {}".format(rank, info)
        print(logging)

def init_seeds(seed,cuda_deterministic=True):
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def train(args):
    # 构建模型及训练数据集
    dist.init_process_group(backend="nccl")
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained)
    if args.short_semantic:
        collate_fn = partial(collate_fn_tag, tokenizer=tokenizer)
        dataset = TemplateSummaryDataset(infor_path=args.infor_file)
        model = SimCSEforShortText(args.pretrained)
        task_name = "TagCL"
    elif args.long_semantic:
        collate_fn = partial(collate_fn_para, tokenizer=tokenizer)
        dataset = InstructionParaDataset(para_path=args.para_file)
        model = SimCSEforLongText(args.pretrained)
        task_name = "ParaCL"

    printlog = partial(print_rank_0, rank=args.logging_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                            sampler=DistributedSampler(dataset))

    model = model.to(int(os.environ["LOCAL_RANK"]))
    model = DDP(model)

    # 创建输出目录
    actual_batch_size = args.batch_size * dist.get_world_size() if dist.is_initialized() else args.batch_size
    add = 'ADD' if args.add_batch else 'NOTADD'
    encoder_name = "roberta" if "roberta" in args.pretrained else "bert"
    major_setting = task_name + "_" + add + '_' + encoder_name + '_lr' + str(args.lr) + '_epochs' + str(
        args.epoch) + '_bs' + str(actual_batch_size) + '_hnweight' + str(args.hard_negative_weight)
    output_dir = os.path.join(args.output_dir, major_setting)
    os.makedirs(output_dir, exist_ok=True)

    # 创建优化器
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'bert' in name or 'mlm' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)
    pretrained_lr = args.lr
    new_lr = pretrained_lr
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': pretrained_lr},
        {'params': new_params, 'lr': new_lr}
    ])
    num_training_steps = args.epoch * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 模型训练
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    step_counter = 0
    for epoch in range(args.epoch):
        dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.to(int(os.environ["LOCAL_RANK"])) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_counter += 1

            if step_counter % args.logging_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                print_rank_0(f'Epoch {epoch}, Step {step_counter}, Loss: {loss.item()}', args.logging_rank)

            progress_bar.update(1)
            progress_bar.set_description_str(f'Epoch{epoch}')
            progress_bar.set_postfix(loss=loss.item())
        
        dist.barrier() #阻塞进程,保证所有的进程都完成了1轮的训练,再进行保存
        if int(os.environ["RANK"]) == args.logging_rank:
            per_epoch_model_output = os.path.join(output_dir, 'epoch' + str(epoch+1))
            printlog("saving model in " + per_epoch_model_output)
            model.module.encoder.save_pretrained(
                per_epoch_model_output, safe_serialization=False
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    rank = torch.distributed.get_rank()
    init_seeds(1 + rank, False)
    train(args)
