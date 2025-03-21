import json
import pickle


name_amazon = {'Instruments': 'Musical_Instruments', 'Pantry': 'Prime_Pantry', 'Arts': 'Arts_Crafts_and_Sewing', 'Office': 'Office_Products', 'Software': 'Software', 'Tools': 'Tools_and_Home_Improvement'}


def extract_meta_data(path):
    meta_data = dict()
    with open(path) as f:
        for line in f.readlines():
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            category = ' '.join(line['category'])
            brand = line['brand']
            title = line['title']

            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = category
            meta_data[asin] = attr_dict
    return meta_data



def data_save_json(train_data, val_data, test_data, smap, dataset_name):
    train_data_f = open('/datasets/' + dataset_name + '/train.json', 'w', encoding='utf8')
    json.dump(train_data, train_data_f)
    train_data_f.close()

    val_data_f = open('/datasets/' + dataset_name + '/val.json', 'w', encoding='utf8')
    json.dump(val_data, val_data_f)
    val_data_f.close()

    test_data_f = open('/datasets/' + dataset_name + '/test.json', 'w', encoding='utf8')
    json.dump(test_data, test_data_f)
    test_data_f.close()

    smap_f = open('/datasets/' + dataset_name + '/smap.json', 'w', encoding='utf8')
    json.dump(smap, smap_f)
    smap_f.close()


    meta_file = '/datasets/' + dataset_name + '/meta_data.json'
    meta_dict = extract_meta_data('/data/' + dataset_name + '/meta_' + name_amazon[dataset_name] + '.json')
    
    meta_f = open(meta_file, 'w', encoding='utf8')
    
    json.dump(meta_dict, meta_f)
    meta_f.close()

def preprocess(data_raw):
    dict_data = {}
    for line_temp in data_raw:
        dict_data[int(line_temp[0])] = [line_temp[1], line_temp[3]]
    return dict_data


def main():
    dataset_name = 'Arts'
    train_data_path = dataset_name + '/user_item_title_negitem_neg_title_seq_train.pkl'
    val_data_path = dataset_name + '/user_item_title_negitem_neg_title_seq_val.pkl'
    test_data_path = dataset_name + '/user_item_title_negitem_neg_title_seq_test.pkl'
    smap = '/dataset/' + dataset_name + '_negative_noaug/' + dataset_name + '_negative_noaug.item2index'

    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    train_data_negative = preprocess(train_data)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f) 
    val_data_negative = preprocess(val_data)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f) 
    test_data_negative = preprocess(test_data)
    dict_smap = {}
    with open(smap, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            dict_smap[temp[0]] = int(temp[1])

    data_save_json(train_data_negative, val_data_negative, test_data_negative, dict_smap, dataset_name)


if __name__ == '__main__':
    main()
