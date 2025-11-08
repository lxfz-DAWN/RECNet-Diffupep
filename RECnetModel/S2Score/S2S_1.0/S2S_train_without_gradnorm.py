import torch
from transformers import Trainer, DefaultDataCollator
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from accelerate import Accelerator
import csv
from S2S_utils import S2S_Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

class CustomTrainer(Trainer):
    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, label_cut, train_target, accelerator, optimizer, max_grad_norm=1.0, scheduler = None):
        super().__init__(model, args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset)
        self.label_cut = label_cut
        self.train_target = train_target
        self.accelerator = accelerator
        self.s2s_loss = S2S_Loss(label_cut, train_target)
        self.optimizer = optimizer
        self.data_collator = data_collator
        self.scaler = accelerator.scaler 
        self.max_grad_norm = max_grad_norm  # 新增：梯度裁剪的最大范数
        self.lr_scheduler = scheduler  # 新增：学习率调度器
        
        self.steps_per_0_1_epoch = int(len(self.train_dataset) / self.args.per_device_train_batch_size * 0.1)
        self.last_valid_step = 0

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels").to(self.accelerator.device)
    #     inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
    #     outputs = model(
    #         ligand=inputs['ligand'],
    #         receptor=inputs['receptor'],
    #         merge=inputs['merge']
    #     )

    #     # 计算每个任务的损失
    #     loss_content = self.s2s_loss.total_loss(outputs, labels)
    #     total_loss = sum(loss_content)  # 直接求和，不再使用GradNorm

        # return (total_loss, outputs) if return_outputs else total_loss
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.amp.autocast(device_type='cuda'): # 添加autocast上下文
            labels = inputs.pop("labels").to(self.accelerator.device)
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss_content = self.s2s_loss.total_loss(outputs, labels)
            total_loss = sum(loss_content)
            # print(f'label:{labels},input{inputs},output:{outputs},total_loss:{total_loss}')
            
        del labels, outputs, loss_content  # 如果这些变量后续不再使用，可以释放
        torch.cuda.empty_cache()  # 释放 GPU 缓存
        
        return (total_loss, outputs) if return_outputs else total_loss

    # def training_step(self, model, inputs, num_items_in_batch):
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     # 清零梯度
    #     self.optimizer.zero_grad()

    #     # 计算损失
    #     loss = self.compute_loss(model, inputs)

    #     # 缩放损失
    #     scaled_loss = self.scaler.scale(loss)

    #     # 手动反向传播
    #     scaled_loss.backward()

    #     # 取消梯度缩放
    #     self.scaler.unscale_(self.optimizer)

    #     # 梯度裁剪
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    #     # 更新模型参数
    #     self.scaler.step(self.optimizer)

    #     # 更新缩放因子
    #     self.scaler.update()

    #     # 更新学习率调度器
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()

    #     return loss.detach()
    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.optimizer.zero_grad()

        # 计算损失（内部已包含autocast）
        loss = self.compute_loss(model, inputs)

        # 使用accelerator处理反向传播
        self.accelerator.backward(loss)

        # 梯度裁剪（仅在梯度同步时执行）
        if self.accelerator.sync_gradients:
            clip_grad_norm_(model.parameters(), self.max_grad_norm)

        # 更新参数和学习率
        # self.optimizer.step()
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        del inputs  # 如果 inputs 后续不再使用，可以释放
        torch.cuda.empty_cache()  # 释放 GPU 缓存
        current_step = self.state.global_step
        # # 每 0.1 个 epoch 检查一次损失值
        
        # if current_step % self.steps_per_0_1_epoch == 0:
        #     loss_value = loss.item()
        #     if torch.isnan(torch.tensor(loss_value)):
        #         # 损失值为 nan，保存前 0.1 个 epoch 的模型
        #         self.save_model(f"checkpoint_step_{self.last_valid_step}")
        #         print(f"Loss became NaN at step {current_step}. Saved model at step {self.last_valid_step}.")
        #         # 可以选择提前终止训练
        #         # self.state.global_step = self.args.max_steps
        #     else:
        #         self.last_valid_step = current_step
        # return loss.detach()

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
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total_loss = 0
        total_samples = 0
        criterion = torch.nn.MSELoss()

        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels")
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 根据你的设计，只对前56个进行MSE计算
                outputs = outputs[:, :56] 
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        eval_loss = total_loss / total_samples
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_runtime": 0,  # 示例值，可根据实际情况修改
            f"{metric_key_prefix}_samples_per_second": 0,  # 示例值，可根据实际情况修改
            f"{metric_key_prefix}_steps_per_second": 0,  # 示例值，可根据实际情况修改
            "epoch": self.state.epoch
        }

        self.log(metrics)
        return metrics
    
    def on_epoch_end(self):
        # 获取当前 epoch 的验证损失
        eval_metrics = self.evaluate()
        val_loss = eval_metrics["eval_loss"]
        # 根据验证损失调整学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)
        
        del eval_metrics  # 如果 eval_metrics 后续不再使用，可以释放
        torch.cuda.empty_cache()
    

        
class CustomDataCollator(DefaultDataCollator):
    def torch_call(self, features):
        batch = {
            "ligand": torch.stack([feature["ligand"] for feature in features]),
            "receptor": torch.stack([feature["receptor"] for feature in features]),
            "merge": torch.stack([feature["merge"] for feature in features]),
            "labels": torch.stack([feature["labels"] for feature in features])
        }
        return batch