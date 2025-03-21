import pandas as pd
import argparse
import os
import html
from collections import Counter
from tqdm import tqdm
import json
import re
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch
import random
import numpy as np


amazon_dataset2fullname = {
    'beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def review_data(args):
    # (item, user, rating, review, time)
    dataset_full_name = amazon_dataset2fullname[args.dataset]
    # path_review = os.path.join(dataset_full_name, dataset_full_name+'.json')
    path_review = os.path.join(args.dataset, dataset_full_name+'_5.json')
    
    review_list = []
    
    with open('/data/' + path_review, 'r') as fp:
        for line in tqdm(fp, desc='Load reviews'):
            data = json.loads(line)
            
            item = data['asin']
            user = data['reviewerID']
            time_stamp = int(data['unixReviewTime'])
            rating = data['overall']
            review_data_temp = ''
            if 'reviewText' in data:
                review_text = clean_text(data['reviewText'])
                if review_text != '.':
                    review_data_temp = review_text + ' '
            summary_data = ''
            if 'summary' in data:
                review_summary = clean_text(data['summary'])
                if review_summary != '.':
                    summary_data = summary_data + review_summary
            review_list.append([item, user, rating, review_data_temp, summary_data, time_stamp])
    
    return review_list
            

def k_core_filter(pd_data, k=5):
    count = 0
    while True:
        if count > 4:
            break
        user_count = Counter(pd_data['user'])
        item_count = Counter(pd_data['item'])
        delete_user = []
        delete_item = []
        for temp in user_count.items():
            if temp[1] < k:
                delete_user.append(temp[0])
        for temp in item_count.items():
            if temp[1] < k:
                delete_item.append(temp[0])
        if len(delete_user) == 0 and len(delete_item) == 0:
            break
        pd_data = pd_data[~pd_data['user'].isin(delete_user)]
        pd_data = pd_data[~pd_data['item'].isin(delete_item)]
        count += 1
    return pd_data


def group_user(pd_data, min_inter=5):
    
    item_encode = pd.factorize(pd_data['item'])
    user_encode = pd.factorize(pd_data['user'])
    
    pd_data['item'] = item_encode[0]
    pd_data['user'] = user_encode[0]
    
    
    list_inters = []
    time_ascending_f = lambda x: x.sort_values(by='time', ascending=True)
    pd_groups = pd_data.groupby('user', as_index=False).apply(time_ascending_f)
    
    for i in range(0, pd_groups.index[-1][0]+1):
        group_temp = pd_groups.loc[i]
        if len(group_temp) >= min_inter:
            group_temp_list = []
            for j in range(len(group_temp)):
                group_temp_list.append(group_temp.iloc[j].tolist())    
            
            list_inters.append(group_temp_list)

    return list_inters, item_encode[1], user_encode[1]


def data_split(data_list, item_encode, user_encode, asin_title_dict):
    item_review_pairs = []
    count = 0
    for user_inter in data_list:
        if len(user_inter[-1][3]) > 0:
            try: 
                item_list = []
                item_review_list = []
                title_list = []
                for temp_record in user_inter:
                    item_list.append(temp_record[0])
                    item_review_list.append(temp_record[3])
                    title_list.append(asin_title_dict[item_encode[temp_record[0]]])
                item_review_pairs.append([count, item_list, title_list, item_review_list])
                count += 1
            except:
                pass
    
    train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris = [], [], [], [], []
    for i, item_review_pairs_temp in enumerate(item_review_pairs):
        if i % 10 < 8:
            train_item_review_pairs.append(item_review_pairs_temp)
            if i % 10 < 4:
                clip_item_review_pairs.append(item_review_pairs_temp)
            else:
                generation_item_review_paris.append(item_review_pairs_temp)
        elif i % 10 == 8:
            val_item_review_pairs.append(item_review_pairs_temp)
        else:
            test_item_review_pairs.append(item_review_pairs_temp)
    return train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris


def random_sample(id, item_encode, asin_title_dict):
    negative_num = 1000
    list_negative = random.sample(range(len(item_encode)), negative_num+1)
    if id in list_negative:
        list_negative.remove(id)
    else:
        list_negative = list_negative[:negative_num]
    list_negative = [id] + list_negative
    title_list_negative = [asin_title_dict[item_encode[i]] for i in list_negative]
    return list_negative, title_list_negative


def data_split_negative_sample_leavle_one_out(data_list, item_encode, user_encode, asin_title_dict):
    item_review_pairs_train, item_review_pairs_val, item_review_pairs_test = [], [], []
    count = 0
    for user_inter in data_list:
        if len(user_inter[-1][3]) > 0:
            try: 
                item_list = []
                item_review_list = []
                title_list = []
                # val_neg_id_list, val_neg_title_list, test_neg_id_list, test_neg_title_list = [], [], [], []
                for temp_record in user_inter:
                    item_list.append(temp_record[0])
                    item_review_list.append(temp_record[3])
                    title_list.append(asin_title_dict[item_encode[temp_record[0]]])
                
                train_neg_id_list, train_neg_title_list = random_sample(item_list[-3], item_encode, asin_title_dict)
                val_neg_id_list, val_neg_title_list = random_sample(item_list[-2], item_encode, asin_title_dict)
                test_neg_id_list, test_neg_title_list = random_sample(item_list[-1], item_encode, asin_title_dict)
                item_review_pairs_train.append([count, item_list[:-3], title_list[:-3], item_review_list[:-3], train_neg_id_list, train_neg_title_list])
                item_review_pairs_val.append([count, item_list[:-2], title_list[:-2], item_review_list[:-2], val_neg_id_list, val_neg_title_list])
                item_review_pairs_test.append([count, item_list[:-1], title_list[:-1], item_review_list[:-1], test_neg_id_list, test_neg_title_list])
                count += 1
            except:
                pass
    return item_review_pairs_train, item_review_pairs_val, item_review_pairs_test 


def data_split_leaveoneout(data_list, item_encode, asin_category_dict, dataset_name):
    data_all_list = [] 
    id_category_dict = {} 
    for user_inter in data_list:
        if len(user_inter) >= 5:
            data_all_list.append([i[0]+1 for i in user_inter])
    data_train = {}
    data_val = {}
    data_test = {}
    umap = {}
    for id_list, list_temp in enumerate(data_all_list):
        data_train[id_list] = list_temp[:-2]
        data_val[id_list] = [list_temp[-2]]
        data_test[id_list] = [list_temp[-1]] 
        umap[id_list+1] = id_list
    smap = {}
    for i in range(len(item_encode)):
        smap[i+1] = i+1
    data_all = {'train': data_train, 'val': data_val, 'test': data_test, 'umap': umap, 'smap': smap}
    
    for id, item in enumerate(item_encode):
        try:
            id_category_dict[id] = asin_category_dict[item]
        except:
            id_category_dict[id] = ["None"]
    
    with open(os.path.join('/data/' + amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_data_all.pkl'), 'wb') as f:
        pickle.dump(data_all, f)
    with open(os.path.join('/data/' + amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_id_category_dict.pkl'), 'wb') as f:
        pickle.dump(id_category_dict, f)
    

def review_seq_save(train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris, dataset_name):
    train_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_review_seq_train.pkl')
    val_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_review_seq_val.pkl')
    test_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_review_seq_test.pkl')
    clip_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_review_seq_clip.pkl')
    generation_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_review_seq_generation.pkl')
    
    with open(train_path_save, 'wb') as f_train:
        pickle.dump(train_item_review_pairs, f_train)
    with open(val_path_save, 'wb') as f_val:
        pickle.dump(val_item_review_pairs, f_val)
    with open(test_path_save, 'wb') as f_test:
        pickle.dump(test_item_review_pairs, f_test)

    with open(clip_path_save, 'wb') as f_test:
        pickle.dump(clip_item_review_pairs, f_test)
    with open(generation_path_save, 'wb') as f_test:
        pickle.dump(generation_item_review_paris, f_test)


def id2title(item_review_pairs, item_encode, asin_title_dict):
    title_item_review_pairs = []
    for paris_temp in item_review_pairs:
        seq_list_id = paris_temp[0]
        review_list = paris_temp[1]
        title_list = []
        try:
            for id_temp in seq_list_id:
                title_list.append(asin_title_dict[item_encode[id_temp]])
            title_item_review_pairs.append([title_list, review_list])
        except:
            pass
    return title_item_review_pairs


def review_title_seq_save(item_encode, asin_title_dict, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris, dataset_name):
    train_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_review_seq_train.pkl')
    val_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_review_seq_val.pkl')
    test_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_review_seq_test.pkl')
    clip_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_review_seq_clip.pkl')
    generation_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_review_seq_generation.pkl')

    with open(train_path_save, 'wb') as f_train:
        pickle.dump(id2title(train_item_review_pairs, item_encode, asin_title_dict), f_train)
    with open(val_path_save, 'wb') as f_val:
        pickle.dump(id2title(val_item_review_pairs, item_encode, asin_title_dict), f_val)
    with open(test_path_save, 'wb') as f_test:
        pickle.dump(id2title(test_item_review_pairs, item_encode, asin_title_dict), f_test)

    with open(clip_path_save, 'wb') as f_test:
        pickle.dump(id2title(clip_item_review_pairs, item_encode, asin_title_dict), f_test)
    with open(generation_path_save, 'wb') as f_test:
        pickle.dump(id2title(generation_item_review_paris, item_encode, asin_title_dict), f_test)


def text_data_info(dataset_name):
    asin_title = {}
    asin_category = {}
    asin_description = {}
    with open(os.path.join('/data/' + dataset_name, 'meta_'+ amazon_dataset2fullname[dataset_name]+'.json'), 'r') as f:
    # with open(os.path.join('/data/' + amazon_dataset2fullname[dataset_name], 'meta_'+amazon_dataset2fullname[dataset_name]+'.json'), 'r') as f:
        for line in tqdm(f, desc='Load reviews'):
            data = json.loads(line)
            
            asin_description[data['asin']] = data['description']
            asin_title[data['asin']] = data['title']
            asin_category[data['asin']] =  data['category']
    return asin_title, asin_category, asin_description


def id2title_rec(item_review_pairs, item_encode, asin_title_dict):
    title_label_id = []
    for paris_temp in item_review_pairs:
        seq_list_id = paris_temp[0][:-1]
        # review_list = paris_temp[1]
        title_list = []
        try:
            for id_temp in seq_list_id:
                title_list.append(asin_title_dict[item_encode[id_temp]])
            title_list.append(paris_temp[0][-1])
            # title_item_review_pairs.append([title_list, review_list])
            title_label_id.append(title_list)
        except:
            pass
    return title_label_id


def title_rec_save(item_encode, asin_title_dict, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, dataset_name):
    train_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_labelid_seq_train.pkl')
    val_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_labelid_seq_val.pkl')
    test_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_labelid_seq_test.pkl')
    
    with open(train_path_save, 'wb') as f_train:
        pickle.dump(id2title_rec(train_item_review_pairs, item_encode, asin_title_dict), f_train)
    with open(val_path_save, 'wb') as f_val:
        pickle.dump(id2title_rec(val_item_review_pairs, item_encode, asin_title_dict), f_val)
    with open(test_path_save, 'wb') as f_test:
        pickle.dump(id2title_rec(test_item_review_pairs, item_encode, asin_title_dict), f_test)


def user_item_title_review_save(dataset_name, asin_title_dict, item_encode, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris):
    id_title_dict = {}
    for item_id, item_temp in enumerate(item_encode):
        try:
            id_title_dict[item_id] = asin_title_dict[item_temp]
        except:
            pass
    
    with open(os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'item_id2title_dict.pkl'), 'wb') as f_id_title_dict:
        pickle.dump(id_title_dict, f_id_title_dict)

    train_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_train.pkl')
    val_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_val.pkl')
    test_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_test.pkl')
    clip_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_clip.pkl')
    generation_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_generation.pkl')

    with open(train_path_save, 'wb') as f_train:
        pickle.dump(train_item_review_pairs, f_train)
    with open(val_path_save, 'wb') as f_val:
        pickle.dump(val_item_review_pairs, f_val)
    with open(test_path_save, 'wb') as f_test:
        pickle.dump(test_item_review_pairs, f_test)

    with open(clip_path_save, 'wb') as f_test:
        pickle.dump(clip_item_review_pairs, f_test)
    with open(generation_path_save, 'wb') as f_test:
        pickle.dump(generation_item_review_paris, f_test)


def user_item_title_review_negative_save(dataset_name, asin_title_dict, item_encode, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs):
    id_title_dict = {}
    for item_id, item_temp in enumerate(item_encode):
        try:
            id_title_dict[item_id] = asin_title_dict[item_temp]
        except:
            pass
    
    with open(os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'itemid2title_dict_leaveoneout.pkl'), 'wb') as f_id_title_dict:
        pickle.dump(id_title_dict, f_id_title_dict)

    train_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_negative_label_leaveoneout_train.pkl')
    val_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_negative_label_leaveoneout_val.pkl')
    test_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_negative_label_leaveoneout_test.pkl')

    with open(train_path_save, 'wb') as f_train:
        pickle.dump(train_item_review_pairs, f_train)
    with open(val_path_save, 'wb') as f_val:
        pickle.dump(val_item_review_pairs, f_val)
    with open(test_path_save, 'wb') as f_test:
        pickle.dump(test_item_review_pairs, f_test)


def title_rep_save(dataset_name):
    encoder_t5 = T5EncoderModel.from_pretrained("../pretrain_model/t5/t5-large").to('cuda')
    tokenizer_t5 = T5Tokenizer.from_pretrained("../pretrain_model/t5/t5-large")
    with open(os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'item_id2title_dict.pkl'), 'rb') as f:
        id_title = pickle.load(f)
    ids = list(id_title.keys())
    titles = list(id_title.values())  ## 79264
    
    batch_size = 152
    num = len(id_title) // batch_size
    left = len(id_title) % batch_size
    
    title_rep_list = []
    for i in range(num):    
        encode_temp = tokenizer_t5(titles[i*batch_size:(i+1)*batch_size], return_tensors='pt', truncation=True, padding='max_length', max_length=42).to('cuda')
        enc_out = encoder_t5(**encode_temp, return_dict=True).last_hidden_state
        rep = torch.gather(enc_out, 1, (torch.sum(encode_temp.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, enc_out.shape[-1]).unsqueeze(1)).squeeze().tolist()
        title_rep_list.append(rep)
    
    if left != 0:
        encode_temp = tokenizer_t5(titles[(num)*batch_size:(num)*batch_size+left], return_tensors='pt', truncation=True, padding='max_length', max_length=42).to('cuda')
        enc_out = encoder_t5(**encode_temp, return_dict=True).last_hidden_state
        rep = torch.gather(enc_out, 1, (torch.sum(encode_temp.attention_mask, dim=-1)-1).unsqueeze(1).repeat(1, enc_out.shape[-1]).unsqueeze(1)).squeeze().tolist()
        title_rep_list.append(rep)
    

    title_rep_list = sum(title_rep_list, [])
    title_reps = np.array(title_rep_list)
    
    pad_rep = title_reps.mean(0).tolist()
    

    list_rep = []
    idx = 0
    for i in range(ids[-1]+1):
        if i in ids:
            list_rep.append(title_reps[idx].tolist())
            idx+=1
        else:
            list_rep.append(pad_rep)
    
    titlerep_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_rep.pkl')
    with open(titlerep_path_save, 'wb') as f:
        pickle.dump(list_rep, f)


def Sim_Jaccard(a, b):
    a_set = set(a)
    b_set = set(b)
    return len(a_set & b_set)/len(a_set | b_set)


def title_sim_review_review_label(label_id_dict, user_item_title_review, user_item_title_review_train):
    list_title_sim_review_review = []
    for data_temp in user_item_title_review:
        seq_temp = data_temp[1]
        label_temp = seq_temp[-1]
        if label_temp in label_id_dict:
            seq_candidates_id = label_id_dict[label_temp]
            sim_temp = []
            for id_temp in seq_candidates_id:
                seq_candidate_temp = user_item_title_review_train[id_temp][1]
                sim_temp.append(Sim_Jaccard(seq_candidate_temp, seq_temp))
            
            if len(sim_temp) > 1:
                top_sim_id = np.argsort(sim_temp)[-2]
                seq_reivew_top_sim = user_item_title_review_train[top_sim_id][-1][-1]
                title = data_temp[2][-1]
                review = data_temp[3][-1]
                pairs = (title + ' // ' + seq_reivew_top_sim, review)
            else:
                title = data_temp[2][-1]
                review = data_temp[3][-1]
                pairs = (title, review)
            list_title_sim_review_review.append(pairs)
        else:
            title = data_temp[2][-1]
            review = data_temp[3][-1]
            list_title_sim_review_review.append(pairs)
    return list_title_sim_review_review


def title_review_explanation(dataset_name):
    train_path_user_item_title_review = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_train.pkl')
    val_path_user_item_title_review = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_val.pkl')
    test_path_user_item_title_review = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_user_item_title_review_seq_test.pkl')
    
    with open(train_path_user_item_title_review, 'rb') as f:
        train_user_item_title_review = pickle.load(f)
    with open(val_path_user_item_title_review, 'rb') as f:
        val_user_item_title_review = pickle.load(f)
    with open(test_path_user_item_title_review, 'rb') as f:
        test_user_item_title_review = pickle.load(f)
    
    seqs = [i[1] for i in train_user_item_title_review]
    label_id_dict = {}
    for idx, seq_temp in enumerate(seqs):
        label_temp = seq_temp[-1]
        if label_temp not in label_id_dict:
            label_id_dict[label_temp] = [idx]
        else:
            label_id_dict[label_temp].append(idx)
        
    train_title_sim_review_review_label_list = title_sim_review_review_label(label_id_dict, train_user_item_title_review, train_user_item_title_review)
    val_title_sim_review_review_label_list = title_sim_review_review_label(label_id_dict, val_user_item_title_review, train_user_item_title_review)
    test_title_sim_review_review_label_list = title_sim_review_review_label(label_id_dict, test_user_item_title_review, train_user_item_title_review)


    train_title_sim_review_label_review_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_sim_review_label_review_pairs_train.pkl')
    val_title_sim_review_label_review_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_sim_review_label_review_pairs_val.pkl')
    test_title_sim_review_label_review_path_save = os.path.join(amazon_dataset2fullname[dataset_name], amazon_dataset2fullname[dataset_name]+'_title_sim_review_label_review_pairs_test.pkl')

    with open(train_title_sim_review_label_review_path_save, 'wb') as f_train:
        pickle.dump(train_title_sim_review_review_label_list, f_train)
    with open(val_title_sim_review_label_review_path_save, 'wb') as f_val:
        pickle.dump(val_title_sim_review_review_label_list, f_val)
    with open(test_title_sim_review_label_review_path_save, 'wb') as f_test:
        pickle.dump(test_title_sim_review_review_label_list, f_test)


def title_review_summary_description_leaveonout_save(title_review_summary_description, path_save):
    train_list, val_list, test_list = [], [], []
    for list_temp in title_review_summary_description:
        train_temp, val_temp, test_temp = [i[:-2] for i in list_temp], [i[:-1] for i in list_temp], list_temp
        train_list.append(train_temp)
        val_list.append(val_temp)
        test_list.append(test_temp)
    
    train_path_save = path_save + '/train.pkl'
    val_path_save = path_save + '/val.pkl'
    test_path_save = path_save + '/test.pkl'

    with open(train_path_save, 'wb') as f:
        pickle.dump(train_list, f)
    with open(val_path_save, 'wb') as f:
        pickle.dump(val_list, f)
    with open(test_path_save, 'wb') as f:
        pickle.dump(test_list, f)


def title_review_description(inter_list, item_encode, asin_description, asin_title):
    title_review_summary_description_list = []

    for record_list in inter_list:  ## ['item', 'user', 'rating', 'review', 'summary', 'time']
        title_list, review_list, summary_list, description_list = [], [], [], []
        for temp_record in record_list:
            item_id = temp_record[0]
            item_code = item_encode[item_id]
            if item_code in asin_title and item_code in asin_description:
                title_temp = asin_title[item_code]
                description_temp = asin_description[item_code]
                review_temp = temp_record[3]
                summary_temp = temp_record[4]
                title_list.append(title_temp)
                review_list.append(review_temp)
                summary_list.append(summary_temp)
                description_list.append(description_temp)
        if len(title_list)>=5:
            title_review_summary_description_list.append([title_list, review_list, summary_list, description_list])
    return title_review_summary_description_list




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='Arts', help='Combination of pre-trained datasets, split by comma')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    args = parser.parse_args()
    datasets = ["Arts"]
    for dataset in datasets:
        args.dataset = dataset
        asin_title_dict, asin_category, asin_description = text_data_info(args.dataset)
        
        review_list = review_data(args)
        pd_review = pd.DataFrame(review_list, columns=['item', 'user', 'rating', 'review', 'summary', 'time'])
        pd_review_k_filter = k_core_filter(pd_review, k=5)
        
        inter_list, item_encode, user_encode = group_user(pd_review_k_filter, min_inter=5)

        title_review_summary_description_list = title_review_description(inter_list, item_encode, asin_description, asin_title_dict)
        title_review_summary_description_leaveonout_save(title_review_summary_description_list, '/data/' + datasets[0] + '/title_review_summary_description/')
        
        
        # data_split_leaveoneout(inter_list, item_encode, asin_category, dataset)
        # quit()

        train_item_review_pairs, val_item_review_pairs, test_item_review_pairs = data_split_negative_sample_leavle_one_out(inter_list, item_encode, user_encode, asin_title_dict)
        user_item_title_review_negative_save(args.dataset, asin_title_dict, item_encode, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs)
        
        # train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris = data_split(inter_list, item_encode, user_encode, asin_title_dict, asin_category)
        
        # review_seq_save(train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris, args.dataset)
        # review_title_seq_save(item_encode, asin_title_dict, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris, args.dataset)
        # title_rec_save(item_encode, asin_title_dict, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, args.dataset)
        # user_item_title_review_save(args.dataset, asin_title_dict, item_encode, train_item_review_pairs, val_item_review_pairs, test_item_review_pairs, clip_item_review_pairs, generation_item_review_paris)
        # title_rep_save(args.dataset)

        # title_review_explanation(args.dataset)



if __name__ == '__main__':
    main()