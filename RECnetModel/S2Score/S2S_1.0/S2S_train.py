import torch
from transformers import Trainer, DefaultDataCollator
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from accelerate import Accelerator
import csv
from S2S_utils import S2S_Loss, GradNormLoss

class CustomTrainer(Trainer):
    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, label_cut, train_target, accelerator, optimizer):
        super().__init__(model, args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset)
        self.label_cut = label_cut
        self.train_target = train_target
        self.accelerator = accelerator
        self.s2s_loss = S2S_Loss(label_cut, train_target)
        self.grad_norm_loss = GradNormLoss(num_of_task=len(train_target))
        self.optimizer = optimizer
        self.data_collator = data_collator
        self.scaler = accelerator.scaler 

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 这里我们使用线性预热调度器作为示例
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.accelerator.device)
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        outputs = model(
            ligand=inputs['ligand'],
            receptor=inputs['receptor'],
            merge=inputs['merge']
            )

        # 计算每个任务的损失
        loss_content = self.s2s_loss.total_loss(outputs, labels)
        loss_content = torch.stack(loss_content)

        # 使用 GradNormLoss 计算加权损失
        total_loss = self.grad_norm_loss(loss_content)
        # print(loss_content,total_loss)

        # with open('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/trying/why_loss_so_big.csv','w', newline = '') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(loss_content , total_loss)
            
        return (total_loss, outputs) if return_outputs else total_loss
    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 清零梯度
        self.optimizer.zero_grad()

        # 计算损失
        loss = self.compute_loss(model, inputs)

        # 缩放损失
        scaled_loss = self.scaler.scale(loss)

        # 手动反向传播并保留计算图
        scaled_loss.backward(retain_graph=True)

        # 执行 GradNorm 的额外前向和反向传播步骤
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            grad_norm_weights = model.module.get_grad_norm_weights()
        else:
            grad_norm_weights = model.get_grad_norm_weights()

        # 手动设置 w 的梯度为零
        self.grad_norm_loss.w.grad = None

        # 执行 GradNorm 的额外前向和反向传播步骤
        self.grad_norm_loss.additional_forward_and_backward(grad_norm_weights, self.optimizer)

        # 更新模型参数
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 更新学习率调度器
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # print(f'Trainer 收到的loss是：{loss}')

        return loss.detach()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        
class CustomDataCollator(DefaultDataCollator):
    def torch_call(self, features):
        batch = {
            "ligand": torch.stack([feature["ligand"] for feature in features]),
            "receptor": torch.stack([feature["receptor"] for feature in features]),
            "merge": torch.stack([feature["merge"] for feature in features]),
            "labels": torch.stack([feature["labels"] for feature in features])
        }
        return batch