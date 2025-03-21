import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,3'
import argparse
from model_utils.exp_model_utils import add_optimizer_args, configure_optimizers, get_total_steps
from dataset_utils.clip_dataset_utils import add_datasets_args, load_data_raw
from dataset_utils.rec_universal_datamodule import UniversalDataModule
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ckpt_utils.exp_universal_checkpoint import UniversalCheckpoint
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from models.t5_adapter import T5Attention_adapter, T5DenseActDense_adapter, T5DenseActDense_adapter_FFT
from models.t5_explain import Fuse_add_encoder_rep
import numpy as np
import copy
from models.evaluate_utils import rouge_score, bleu_score


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)


class Collator_exp_gep():
    def __init__(self, args):
        self.args = args
        self.seq_item = 10
        self.exp_review_len = 128
        self.tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5/t5-small')
        self.title_seq = args.title_seq

    def __call__(self, inputs):
        examples = []
        for input_temp in inputs:  ## title, review, summary, description
            example = {}
            title_list = input_temp[0]
            title_list = title_list[-self.seq_item:]
            title_token = {'input_ids': [], 'attention_mask': []}
            for title_temp in title_list[:-1]:
                title_temp = 'cls ' + title_temp
                token_temp = self.tokenizer(title_temp, truncation=True, padding='max_length', max_length=self.title_seq)                
                cut = sum(token_temp['attention_mask'])
                title_token['input_ids'] += token_temp['input_ids'][:cut]
                title_token['attention_mask'] += token_temp['attention_mask'][:cut]
                
            ## batch size > 1
            title_token['input_ids'] = torch.tensor(title_token['input_ids'] + ((self.seq_item-1)*self.title_seq - len(title_token['input_ids']))*[0])
            title_token['attention_mask'] = torch.tensor(title_token['attention_mask'] + ((self.seq_item-1)*self.title_seq  - len(title_token['attention_mask']))*[0])
            
            ### batch size = 1
            # title_token['input_ids'] = torch.tensor(title_token['input_ids'])
            # title_token['attention_mask'] = torch.tensor(title_token['attention_mask'])
            
            example['title_seq_token'] = title_token
            exp_input = {'input_ids': [], 'attention_mask': []}
            title_last_tokenizer = self.tokenizer(title_list[-1], truncation=True, padding='max_length', max_length=self.title_seq)
            try:
                des_temp = input_temp[-1][-1][0]
            except:
                des_temp = ''
            description_last_tokenizer = self.tokenizer(des_temp, truncation=True, padding='max_length', max_length=self.exp_review_len)
            input_ids_exp_input = (title_last_tokenizer['input_ids'] + description_last_tokenizer['input_ids'])[:self.exp_review_len]
            attention_mask_exp_input = (title_last_tokenizer['attention_mask'] + description_last_tokenizer['attention_mask'])[:self.exp_review_len]
            exp_input['input_ids'] = torch.tensor(input_ids_exp_input + (self.exp_review_len-len(input_ids_exp_input))*[0])
            exp_input['attention_mask'] = torch.tensor(attention_mask_exp_input + (self.exp_review_len-len(attention_mask_exp_input))*[0])
            example['exp_description'] = exp_input
            
            example['review'] = self.tokenizer(input_temp[-2][-1], truncation=True, padding='max_length', max_length=self.exp_review_len, return_tensors="pt")
            
            examples.append(example)
        return default_collate(examples)


class Rec_seq_exp(LightningModule):
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Title_seq=15, item_num=10, small lr=0.0005')
        parser.add_argument('--dataset_name', default='Instruments')
        parser.add_argument('--hidden_size', type=int, default=512)  ## T5-small: 512, T5-base: 768, T5-large: 1024
        parser.add_argument('--title_seq', type=int, default=15)
        parser.add_argument('--encoder_ckpt_path', default=None)
        parser.add_argument('--seq_items', type=int, default=10)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5/t5-small')
        self.encoder = T5EncoderModel.from_pretrained('./pretrain_model/t5/t5-small')
        self.exp_gen = T5ForConditionalGeneration.from_pretrained('./pretrain_model/t5/t5-small')

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
            i.layer[1].DenseReluDense = T5DenseActDense_adapter_FFT(config=encoder_config, len_seq=args.title_seq*(args.seq_items), len_title=args.title_seq)
            
            for name_temp in temp_ffn.named_parameters():
                exec('i.layer[1].DenseReluDense.{} = temp_ffn.{}'.format(name_temp[0], name_temp[0]))
        
        encoder_weight = torch.load(args.encoder_ckpt_path)['state_dict']
        for key_temp in list(encoder_weight.keys()):
            encoder_weight[key_temp.replace('encoder.', '', 1)] = encoder_weight.pop(key_temp)
        self.encoder.load_state_dict(encoder_weight)
        
        self.exp_gen.encoder = self.encoder.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.exp_gen.parameters():
            param.requires_grad = False
        
        
        exp_gen_config = copy.deepcopy(self.exp_gen.config)
        exp_gen_config.use_cache = False
        exp_gen_config.is_encoder_decoder = False
        
        for i in self.exp_gen.decoder.block: 
            """
            temp_selfatt = i.layer[0].SelfAttention
            try: 
                i.layer[0].SelfAttention.relative_attention_bias
                hashas_relative_attention_bias = True
            except:
                hashas_relative_attention_bias = False
            i.layer[0].SelfAttention = T5Attention_adapter(config=exp_gen_config, has_relative_attention_bias=hashas_relative_attention_bias)
            for name_temp in temp_selfatt.named_parameters():
                exec('i.layer[0].SelfAttention.{} = temp_selfatt.{}'.format(name_temp[0], name_temp[0]))
            """
            temp_ffn = i.layer[2].DenseReluDense
            i.layer[2].DenseReluDense = T5DenseActDense_adapter_FFT(config=exp_gen_config, len_seq=args.title_seq*(10-1), len_title=args.title_seq)
            # i.layer[1].DenseReluDense = T5DenseActDense_adapter(config=encoder_config)
            for name_temp in temp_ffn.named_parameters():
                exec('i.layer[2].DenseReluDense.{} = temp_ffn.{}'.format(name_temp[0], name_temp[0]))
            
        
        for param in self.exp_gen.decoder.block[-1].parameters():  ## small: 5, base:11
            param.requires_grad = True
        
        ## decoder fine-tuning full parameter fine-tuning
        # for param in self.exp_gen.decoder.parameters():  ## small: 5, base:11
          #   param.requires_grad = True
            
        self.title_seq = args.title_seq
        self.PPF = Fuse_add_encoder_rep(args.hidden_size)
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.save_hyperparameters(args)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        self.exp_gen.train()
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep_seq = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)
        rep_description_out = self.exp_gen.encoder(**batch['exp_description'], return_dict=True)
        ppf_rep = self.PPF(rep_seq, rep_description_out.last_hidden_state, mask=(batch['exp_description']['input_ids']>0).float().to(rep_seq.device))
        rep_description_out.last_hidden_state = ppf_rep
        
        labels = batch['review']['input_ids'].to(rep_seq.device).squeeze(1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        out_exp = self.exp_gen(input_ids=batch['exp_description']['input_ids'].to(rep_seq.device), labels=labels, encoder_outputs=rep_description_out, attention_mask=batch['exp_description']['attention_mask'].to(rep_seq.device), decoder_attention_mask=batch['review']['attention_mask'].to(rep_seq.device).squeeze(1))
        loss = out_exp.loss
        
        self.log("train_loss", loss.item(), on_epoch=False, prog_bar=True, logger=True)
        if self.trainer.global_rank == 0 and self.global_step == 100:
            report_memory('Seq exp gen')
        return {"loss": loss}
    
   
    def validation_step(self, batch, batch_idx):
        self.exp_gen.eval()
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep_seq = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)
        rep_description_out = self.exp_gen.encoder(**batch['exp_description'], return_dict=True)
        # rep_description_out = self.encoder(**batch['exp_description'], return_dict=True)
        ppf_rep = self.PPF(rep_seq, rep_description_out.last_hidden_state, mask=(batch['exp_description']['input_ids']>0).float().to(rep_seq.device))
        rep_description_out.last_hidden_state = ppf_rep
        
        out_exp = self.exp_gen.generate(input_ids=batch['exp_description']['input_ids'].to(rep_seq.device), attention_mask=batch['exp_description']['attention_mask'].to(rep_seq.device), encoder_outputs=rep_description_out)
        # out_exp = self.exp_gen_r.generate(input_ids=batch['exp_description']['input_ids'].to(rep_seq.device), attention_mask=batch['exp_description']['attention_mask'].to(rep_seq.device))
        generation_exp = self.tokenizer.batch_decode(out_exp, skip_special_tokens=True)
        reviews = self.tokenizer.batch_decode(batch['review']['input_ids'].squeeze(1), skip_special_tokens=True)
        new_tokens_generate = [l.split() for l in generation_exp]
        new_tokens_reference = [ll.split() for ll in reviews]
        
        BLEU1 = bleu_score(new_tokens_reference, new_tokens_generate, n_gram=1, smooth=False)
        BLEU4 = bleu_score(new_tokens_reference, new_tokens_generate, n_gram=4, smooth=False)
        ROUGE = rouge_score(reviews, generation_exp)
        
        metrics = {}
        metrics['BLEU-1'] = BLEU1
        metrics['BLEU-4'] = BLEU4
        metrics['ROUGE-1/f'] = ROUGE['rouge_1/f_score']
        metrics['ROUGE-1/r'] = ROUGE['rouge_1/r_score']
        metrics['ROUGE-1/p'] = ROUGE['rouge_1/p_score']
        metrics['ROUGE-2/f'] = ROUGE['rouge_2/f_score']
        metrics['ROUGE-2/r'] = ROUGE['rouge_2/r_score']
        metrics['ROUGE-2/p'] = ROUGE['rouge_2/p_score']
        metrics['ROUGE-l/f'] = ROUGE['rouge_l/f_score']
        metrics['ROUGE-l/r'] = ROUGE['rouge_l/r_score']
        metrics['ROUGE-l/p'] = ROUGE['rouge_l/p_score']
        return {"metrics":  metrics}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        print('validation_epoch_end')
        metrics_all = self.all_gather(validation_step_outputs)
        val_metrics_dict = {'BLEU-1': [], 'BLEU-4': [], 'ROUGE-1/f': [], 'ROUGE-1/r': [], 'ROUGE-1/p': [], 'ROUGE-2/f': [], 'ROUGE-2/r': [], 'ROUGE-2/p': [], 'ROUGE-l/f': [], 'ROUGE-l/r': [], 'ROUGE-l/p': []}
        val_metrics_dict_mean = {}
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                val_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in val_metrics_dict.items():
            values_mean = round(np.mean(values_temp), 4)
            val_metrics_dict_mean[key_temp] = values_mean
        print(val_metrics_dict_mean)
        self.log("Val_Metrics", val_metrics_dict_mean)
        self.log("BLEU-1", val_metrics_dict_mean['BLEU-1'])
    
    def test_step(self,  batch, batch_idx):
        self.exp_gen.eval()
        seq_titles = batch['title_seq_token']
        last_hidden_state = self.encoder(**seq_titles).last_hidden_state
        rep_seq = torch.gather(last_hidden_state, 1, (torch.sum(seq_titles['attention_mask'], dim=-1)-1).unsqueeze(1).repeat(1, last_hidden_state.shape[-1]).unsqueeze(1)).squeeze(1)
        rep_description_out = self.exp_gen.encoder(**batch['exp_description'], return_dict=True)
        ppf_rep = self.PPF(rep_seq, rep_description_out.last_hidden_state, mask=(batch['exp_description']['input_ids']>0).float().to(rep_seq.device))
        rep_description_out.last_hidden_state = ppf_rep
        
        out_exp = self.exp_gen.generate(input_ids=batch['exp_description']['input_ids'].to(rep_seq.device), attention_mask=batch['exp_description']['attention_mask'].to(rep_seq.device), encoder_outputs=rep_description_out)
        generation_exp = self.tokenizer.batch_decode(out_exp, skip_special_tokens=True)
        reviews = self.tokenizer.batch_decode(batch['review']['input_ids'].squeeze(1), skip_special_tokens=True)
        new_tokens_generate = [l.split() for l in generation_exp]
        new_tokens_reference = [ll.split() for ll in reviews]
        
        BLEU1 = bleu_score(new_tokens_reference, new_tokens_generate, n_gram=1, smooth=False)
        BLEU4 = bleu_score(new_tokens_reference, new_tokens_generate, n_gram=4, smooth=False)
        ROUGE = rouge_score(reviews, generation_exp)
        
        metrics = {}
        metrics['BLEU-1'] = BLEU1
        metrics['BLEU-4'] = BLEU4
        metrics['ROUGE-1/f'] = ROUGE['rouge_1/f_score']
        metrics['ROUGE-1/r'] = ROUGE['rouge_1/r_score']
        metrics['ROUGE-1/p'] = ROUGE['rouge_1/p_score']
        metrics['ROUGE-2/f'] = ROUGE['rouge_2/f_score']
        metrics['ROUGE-2/r'] = ROUGE['rouge_2/r_score']
        metrics['ROUGE-2/p'] = ROUGE['rouge_2/p_score']
        metrics['ROUGE-l/f'] = ROUGE['rouge_l/f_score']
        metrics['ROUGE-l/r'] = ROUGE['rouge_l/r_score']
        metrics['ROUGE-l/p'] = ROUGE['rouge_l/p_score']
        return {"metrics":  metrics}
    
    def test_epoch_end(self, test_step_outputs) -> None:
        print('test_epoch_end')
        metrics_all = self.all_gather(test_step_outputs)
        test_metrics_dict = {'BLEU-1': [], 'BLEU-4': [], 'ROUGE-1/f': [], 'ROUGE-1/r': [], 'ROUGE-1/p': [], 'ROUGE-2/f': [], 'ROUGE-2/r': [], 'ROUGE-2/p': [], 'ROUGE-l/f': [], 'ROUGE-l/r': [], 'ROUGE-l/p': []}
        test_metrics_dict_mean = {}
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                test_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp), 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print(test_metrics_dict_mean)

def main():
    print("t5 base decoder parameter efficient fine-tuning")
    args_parser = argparse.ArgumentParser()
    args_parser = add_optimizer_args(args_parser)
    args_parser = add_datasets_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = Rec_seq_exp.add_module_specific_args(args_parser)

    custom_parser = [
        '--datasets_path_train', 'data/Instruments/title_review_summary_description/train.pkl',
        '--datasets_path_test', 'data/Instruments/title_review_summary_description/test.pkl',
        '--datasets_path_val', 'data/Instruments/title_review_summary_description/val.pkl',
        '--train_batchsize', '256',
        '--val_batchsize', '256',
        '--test_batchsize', '256',
        '--title_seq', '15',
        '--max_epochs', '250',
        '--random_seed', '72',
        '--dataset_name', 'Instruments',
        '--encoder_ckpt_path', 'ckpt/rec/instruments_title_rep_rec_inner_prod_negpredict_unisrec_data_t5_small_seqlen_10_patinence_5_check_epoch_1_title_len_15_ffn_filter_last_onlyseq_weight_qk_adapterdot_real_last_t5block_train_cls_cut_50epoch_bestsave/last.ckpt',
    ]
    
    args = args_parser.parse_args(args=custom_parser)
    
    print(args)
    datasets = load_data_raw(args)
    checkpoint_callback = UniversalCheckpoint(args)

    collate_fn = Collator_exp_gep(args)
    datamodule = UniversalDataModule(collate_fn=collate_fn, args=args, datasets=datasets)
    
    early_stop_callback_step = EarlyStopping(monitor='BLEU-1', min_delta=0.00, patience=2, verbose=False, mode='max')
    # trainer = Trainer(devices=1, accelerator="gpu", strategy = DDPStrategy(find_unused_parameters=False), callbacks=[checkpoint_callback, early_stop_callback], max_epochs=args.max_epochs, check_val_every_n_epoch=10)
    trainer = Trainer(devices=4, accelerator="gpu", strategy = DDPStrategy (find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=args.max_epochs,  check_val_every_n_epoch=1)
    model = Rec_seq_exp(args)
    
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    

if __name__ =="__main__":
    main()


