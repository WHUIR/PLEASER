import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
from model_utils.rec_model_utils import add_optimizer_args, configure_optimizers, get_total_steps
from dataset_utils.clip_dataset_utils import add_datasets_args, load_data_raw
from dataset_utils.rec_universal_datamodule import UniversalDataModule
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ckpt_utils.rec_universal_checkpoint import UniversalCheckpoint
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from transformers import T5Tokenizer, T5EncoderModel
from models.t5_adapter import T5Attention_adapter, T5DenseActDense_adapter_FFT
import numpy as np
import json
import copy


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)


def cal_recall(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def recalls_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_recall(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['Recall@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def calculate_metrics(scores, labels, metric_ks):
    metrics = recalls_and_ndcgs_k(scores, labels, metric_ks)
    return metrics


class Collator():
    def __init__(self, args):
        self.args = args
        self.seq_len = 200
        self.seq_item = 10
        self.tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5/t5-small')
        self.title_seq = args.title_seq

    def __call__(self, inputs):
        examples = []
        for input_temp in inputs:  ## user, item_seq, title_seq, neg_items (pos+neg), neg_titles (pos+neg)
            example = {}
            title_list = input_temp[2]
            title_list = title_list[-self.seq_item:]
            title_token = {'input_ids': [], 'attention_mask': []}
            for title_temp in title_list:
                title_temp = 'cls ' + title_temp
                token_temp = self.tokenizer(title_temp, truncation=True, padding='max_length', max_length=self.title_seq)                
                cut = sum(token_temp['attention_mask'])
                title_token['input_ids'] += token_temp['input_ids'][:cut]
                title_token['attention_mask'] += token_temp['attention_mask'][:cut]

            ## batch size > 1
            title_token['input_ids'] = torch.tensor(title_token['input_ids'] + (self.seq_item*self.title_seq - len(title_token['input_ids']))*[0])
            title_token['attention_mask'] = torch.tensor(title_token['attention_mask'] + (self.seq_item*self.title_seq  - len(title_token['attention_mask']))*[0])
            
            ### batch size = 1
            # title_token['input_ids'] = torch.tensor(title_token['input_ids'])
            # title_token['attention_mask'] = torch.tensor(title_token['attention_mask'])
            
            example['title_seq_token'] = title_token
            
            ## negative_samples
            neg_title_list = []
            for neg_title in input_temp[-1]:
                title_temp = 'cls ' + neg_title
                neg_title_list.append(title_temp)
            example['neg_titles'] = neg_title_list
            examples.append(example)
        return default_collate(examples)


class Rec_seq(LightningModule):
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Title_seq=15, item_num=10, small lr=0.0005')
        parser.add_argument('--dataset_name', default='Office')
        parser.add_argument('--hidden_size', type=int, default=512)  ## T5-small: 512, T5-base: 768, T5-large: 1024
        parser.add_argument('--title_seq', type=int, default=15) 
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5/t5-small')
        self.encoder = T5EncoderModel.from_pretrained('./pretrain_model/t5/t5-small')
        
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        # for param in self.encoder.encoder.block[-1].parameters(): 
        #     param.requires_grad = True
        
        encoder_config = copy.deepcopy(self.encoder.config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        for i in self.encoder.encoder.block:
            temp_selfatt = i.layer[0].SelfAttention
            try: 
                i.layer[0].SelfAttention.relative_attention_bias
                hashas_relative_attention_bias = True
            except:
                hashas_relative_attention_bias = False
            i.layer[0].SelfAttention = T5Attention_adapter(config=encoder_config, has_relative_attention_bias=hashas_relative_attention_bias)
            for name_temp in temp_selfatt.named_parameters():
                exec('i.layer[0].SelfAttention.{} = temp_selfatt.{}'.format(name_temp[0], name_temp[0]))
            temp_ffn = i.layer[1].DenseReluDense
            i.layer[1].DenseReluDense = T5DenseActDense_adapter_FFT(config=encoder_config, len_seq=args.title_seq*10, len_title=args.title_seq)
            for name_temp in temp_ffn.named_parameters():
                exec('i.layer[1].DenseReluDense.{} = temp_ffn.{}'.format(name_temp[0], name_temp[0]))
            
            
        self.title_seq = args.title_seq
        self.loss_ce = nn.CrossEntropyLoss()
        self.save_hyperparameters(args)
    
    def load_title(self, path_title):
        with open(path_title, 'r') as f:
            title_dict = json.load(f)
        titles = list(title_dict.values())
        titles_token = self.tokenizer(titles, return_tensors='pt', truncation=True, padding='max_length', max_length=self.title_seq)
        return titles_token

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        return configure_optimizers(self)
        
    def training_step(self, batch, batch_idx):
        
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)

        """
        ### Full predicted 
        label_hidden = self.encoder(**self.all_title_token.to(rep.device)).last_hidden_state
        rep_label = torch.gather(label_hidden, 1, (torch.sum(self.all_title_token.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## all_items x hidden_size
        scores = torch.matmul(rep, rep_label.transpose(0, 1))
        label = batch['label_id'].to(scores.device)
        loss = self.loss_ce(scores, label.long())
        """
        
        ### negative samples
        neg_titles_list = np.array(batch['neg_titles']).transpose().tolist()  ## batch_size x (1+neg_nums)
        label_ids = self.tokenizer(sum(neg_titles_list, []), return_tensors='pt', truncation=True, padding='max_length', max_length=self.title_seq).to(self.encoder.device)
        label_hidden = self.encoder(**label_ids).last_hidden_state
        
        rep_label = torch.gather(label_hidden, 1, (torch.sum(label_ids.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## batchx(1+neg_nums), hidden_size
        rep_label = rep_label.reshape(rep.shape[0], -1, rep.shape[-1])
        scores = torch.matmul(rep.unsqueeze(1), rep_label.transpose(1, 2)).squeeze(1)

        label = torch.zeros(scores.shape[0]).to(scores.device)
        loss = self.loss_ce(scores, label.long())
        
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        if self.trainer.global_rank == 0 and self.global_step == 100:
            report_memory('Seq rec')
        return {"loss": loss}
    
   
    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)
        
        """
        ### full predicted
        label_hidden = self.encoder(**self.all_title_token.to(rep.device)).last_hidden_state
        rep_label = torch.gather(label_hidden, 1, (torch.sum(self.all_title_token.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## all_items x hidden_size
        scores = torch.matmul(rep, rep_label.transpose(0, 1))
        labels = batch['label_id'].to(scores.device)
        """
        
        ### negative sample
        neg_titles_list = np.array(batch['neg_titles']).transpose().tolist()  ## batch_size x 1+neg_nums
        label_ids = self.tokenizer(sum(neg_titles_list, []), return_tensors='pt', truncation=True, padding='max_length', max_length=self.title_seq).to(self.encoder.device)
        label_hidden = self.encoder(**label_ids).last_hidden_state
        
        rep_label = torch.gather(label_hidden, 1, (torch.sum(label_ids.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## batchx(1+neg_nums), hidden_size
        rep_label = rep_label.reshape(rep.shape[0], -1, rep.shape[-1])
        
        scores = torch.matmul(rep.unsqueeze(1), rep_label.transpose(1, 2)).squeeze(1)
        labels = torch.zeros(scores.shape[0]).to(scores.device)
        metrics = calculate_metrics(scores, labels.unsqueeze(1), metric_ks=[5, 10, 20, 50])
        
        return {"metrics":  metrics}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        print('validation_epoch_end')
        metrics_all = self.all_gather(validation_step_outputs)
        val_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        val_metrics_dict_mean = {}
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                val_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in val_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            val_metrics_dict_mean[key_temp] = values_mean
        print(val_metrics_dict_mean)
        self.log("Val_Metrics", val_metrics_dict_mean)
        self.log("Recall@10", val_metrics_dict_mean['Recall@10'])
    
    def test_step(self,  batch, batch_idx):
        self.encoder.eval()
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)

        """
        ### full predict
        label_hidden = self.encoder(**self.all_title_token.to(rep.device)).last_hidden_state
        rep_label = torch.gather(label_hidden, 1, (torch.sum(self.all_title_token.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## all_items x hidden_size
        scores = torch.matmul(rep, rep_label.transpose(0, 1))
        label_ids = batch['label_id'].to(scores.device)
        """
        
        ### negative sample
        neg_titles_list = np.array(batch['neg_titles']).transpose().tolist()  ## batch_size x 1+neg_nums
        label_ids = self.tokenizer(sum(neg_titles_list, []), return_tensors='pt', truncation=True, padding='max_length', max_length=self.title_seq).to(self.encoder.device)
        label_hidden = self.encoder(**label_ids).last_hidden_state
        rep_label = torch.gather(label_hidden, 1, (torch.sum(label_ids.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, label_hidden.shape[-1]).unsqueeze(1)).squeeze(1)  ## batchx(1+neg_nums), hidden_size
        rep_label = rep_label.reshape(rep.shape[0], -1, rep.shape[-1])
        
        scores = torch.matmul(rep.unsqueeze(1), rep_label.transpose(1, 2)).squeeze(1)
        label_ids = torch.zeros(scores.shape[0]).to(scores.device)
        
        metrics = calculate_metrics(scores, label_ids.unsqueeze(1), metric_ks=[5, 10, 20, 50])
        return {"metrics":  metrics}
    
    def test_epoch_end(self, test_step_outputs) -> None:
        print('test_epoch_end')
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        test_metrics_dict_mean = {}
        metrics_all = self.all_gather(test_step_outputs)
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                test_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print(test_metrics_dict_mean)
        
        


def main():
    print("lr=0.0005,title_len=15,small, emb, adapter with FFN_filter, adapter_qk_dot and full parameter fine-tuning cls cut, val is test")
    args_parser = argparse.ArgumentParser()
    args_parser = add_optimizer_args(args_parser)
    args_parser = add_datasets_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = Rec_seq.add_module_specific_args(args_parser)

    custom_parser = [
        '--datasets_path_train', 'data/Office/user_item_title_negitem_neg_title_seq_train.pkl',
        '--datasets_path_test', 'data/Office/user_item_title_negitem_neg_title_seq_test.pkl',
        '--datasets_path_val', 'data/Office/user_item_title_negitem_neg_title_seq_val.pkl',
        '--train_batchsize', '5',
        '--val_batchsize', '5',
        '--test_batchsize', '5',
        '--title_seq', '15',
        '--max_epochs', '50',
        '--random_seed', '72',
        '--dataset_name', 'Office',
    ]
    

    args = args_parser.parse_args(args=custom_parser)
    
    print(args)
    datasets = load_data_raw(args)
    
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)
    collate_fn = Collator(args)
    datamodule = UniversalDataModule(collate_fn=collate_fn, args=args, datasets=datasets)
    
    early_stop_callback_step = EarlyStopping(monitor='Recall@10', min_delta=0.00, patience=3, verbose=False, mode='max')
    trainer = Trainer(devices=4, accelerator="gpu", strategy = DDPStrategy (find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=args.max_epochs,  check_val_every_n_epoch=1)
    # trainer = Trainer(devices=1, accelerator="gpu", strategy = DDPStrategy(find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=args.max_epochs, check_val_every_n_epoch=1)
    model = Rec_seq(args)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    

if __name__ =="__main__":
    main()


