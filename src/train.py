from dataset_fn import *
from model_class import *
import argparse
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.optim as optim
import os
from functools import partial

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
    parser.add_argument("--hard_negative_weight",type=float, default=0, help='给每个困难负例增加权重，提高任务难度')
    args = parser.parse_args()
    return args


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
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained)
    if args.short_semantic:
        collate_fn = partial(collate_fn_tag, tokenizer=tokenizer)
        dataset = TemplateSummaryDataset(infor_path=args.infor_file)
        model = SimCSEforShortText(args.pretrained)
        task_name = "TagCL"
    elif args.long_semantic:
        collate_fn = partial(collate_fn_para, tokenizer=tokenizer)
        dataset = InstructionParaDataset(para_path=args.para_file)
        model = SimCSEforLongText(pretrained=args.pretrained, hard_negative_weight=args.hard_negative_weight,
                add_batch=args.add_batch)
        task_name = "ParaCL"

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    model.train()

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
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    step_counter = 0
    for epoch in range(args.epoch):
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_counter += 1
            if step_counter % args.logging_step == 0:
                print(f'Epoch {epoch}, Step {step_counter}, Loss: {loss.item()}')

            progress_bar.update(1)
            progress_bar.set_description_str(f'Epoch{epoch}')
            progress_bar.set_postfix(loss=loss.item())

        if args.add_batch:
            add = 'ADD'
        else:
            add = 'NOTADD'

        encoder_name = "roberta" if "roberta" in args.pretrained else "bert"
        output_dir = os.path.join(args.output_dir, encoder_name)
        os.makedirs(args.output_dir, exist_ok=True)

        model_output_name =  task_name + "_"  + add + '_' + encoder_name + \
                '_lr' + str(args.lr) + '_epoch' + str(epoch) + '_hnweight' + str(args.hard_negative_weight)
        print(model_output_name)
        model.encoder.save_pretrained(
            os.path.join(output_dir,model_output_name),safe_serialization=False
            )
        


if __name__ == "__main__":
    args = parse_args()
    init_seeds(0, False)
    train(args)
