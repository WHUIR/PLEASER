import json
import pickle


def title_save(data_path_load, data_path_save, title_dict, item_dict):
    lines_title = []
    with open(data_path_load, 'r') as f:
        lines = f.readlines()
        for line_temp in lines[1:]:
            lines_data = line_temp.split('\t')
            user_id = int(lines_data[0])
            item_seq = [int(i) for i in lines_data[1].split(' ')]
            title_seq = [title_dict[item_dict[i]] for i in item_seq]
            negs = lines_data[3].replace('\n', '').split(' ')
            if negs[-1] == '':
                negs = negs[:-1] 
            negative_item = [int(lines_data[2])] + [int(i) for i in negs]
            
            negative_title = [title_dict[item_dict[i]] for i in negative_item]
            lines_title.append([user_id, item_seq, title_seq, negative_item, negative_title])
    
    with open(data_path_save, 'wb') as fw:
        pickle.dump(lines_title, fw)


def main():
    dataset_name = "Arts"
    path_train = '/dataset/' + dataset_name + '_negative_noaug' + '/' + dataset_name + '_negative_noaug.train.inter'
    path_val = '/dataset/' + dataset_name + '_negative_noaug' + '/' + dataset_name + '_negative_noaug.valid.inter'
    path_test = '/dataset/' + dataset_name + '_negative_noaug' + '/' + dataset_name + '_negative_noaug.test.inter'
    path_item_dict= '/dataset/' + dataset_name + '_negative_noaug' + '/' + dataset_name + '_negative_noaug.item2index'
    path_item_title_dict = '/data/' + dataset_name + '/stmap.pkl'
    
    path_train_title_save =  '/data/' + dataset_name + '/user_item_title_negitem_neg_title_seq_train.pkl'
    path_val_title_save =  '/data/' + dataset_name + '/user_item_title_negitem_neg_title_seq_val.pkl'
    path_test_title_save =  '/data/' + dataset_name + '/user_item_title_negitem_neg_title_seq_test.pkl'

    item_dict = {}
    with open(path_item_dict, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            item_dict[int(temp[1])] = temp[0]
            
    
    # with open(path_title_dict_recformer, 'r') as f:
    #     title_dict = json.load(f)
    # with open(path_item_dict_recformer, 'r') as f:
    #     item_code_dict = json.load(f)
    # item_dict = {}
    # for temp in item_code_dict.items():
    #     item_dict[temp[1]] = temp[0]
    # title_item_dict = {}
    # for item_id in title_dict:
    #     title_item_dict[item_dict[int(item_id)]] = title_dict[item_id]
    with open(path_item_title_dict, 'rb') as f:
        title_item_dict = pickle.load(f)
    
    
    title_save(path_train, path_train_title_save, title_item_dict, item_dict)
    title_save(path_val, path_val_title_save, title_item_dict, item_dict)
    title_save(path_test, path_test_title_save, title_item_dict, item_dict)



if __name__ == '__main__':
    main()

