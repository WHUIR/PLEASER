import os
import json
import pickle


amazon_dataset2fullname = {
    'beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'books': 'Books',
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
    'instrument': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}


def load_meta_json(path_meta):
    product_description = {}
    with open(path_meta, 'r') as f:
        for line in f:
            data = json.loads(line)
            id_product = data['asin']
            description_product = data['description']
            product_description[id_product] = description_product 
    return product_description


def load_record(path_review):
    product_review = {}
    with open(path_review, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'reviewText' in data.keys():
                review_text = data['reviewText']
                id_product = data['asin']
                # id_user = data['reviewerID']
                product_review[id_product] = review_text
    return product_review


def merge_desp_review(dict_desp, dict_review):
    prod_desp_review = {}
    for prod_id in dict_desp:
        if prod_id in dict_review:
            if len(dict_desp[prod_id]) > 0 and len(dict_review[prod_id]) > 0:
                sec = dict_desp[prod_id][0] + dict_review[prod_id]
            elif len(dict_desp[prod_id]) > 0 and len(dict_review[prod_id]) == 0:
                sec = dict_desp[prod_id][0]
            elif len(dict_desp[prod_id]) == 0 and len(dict_review[prod_id]) > 0:
                sec = dict_review[prod_id]
            prod_desp_review[prod_id] = sec
        else:
            if len(dict_desp[prod_id]) > 0:
                sec = dict_desp[prod_id][0]
                prod_desp_review[prod_id] = sec
    return prod_desp_review


def data_save(save_path, dict_save):
    with open(save_path, 'wb') as f:
        pickle.dump(dict_save, f)


def main():
    dataset = 'instrument'
    ori_name = amazon_dataset2fullname[dataset]
    path_meta = os.path.join(dataset, f'meta_{ori_name}.json') 
    prod_descrip = load_meta_json(path_meta)
    path_review = os.path.join(dataset, ori_name+'.json')
    prod_review = load_record(path_review)
    prod_desp_review = merge_desp_review(prod_descrip, prod_review)
    path_save_prod_desp_review = os.path.join(dataset, ori_name+'.pkl')
    data_save(path_save_prod_desp_review, prod_desp_review)


if __name__ == '__main__':
    main()
