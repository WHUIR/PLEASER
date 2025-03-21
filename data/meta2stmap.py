import json
import pickle


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



def smap_tmap_extract(path_data, data_save):
    title_dict = {}
    with open(path_data, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = json.loads(line)
            title_dict[line['asin']] = line['title']
    print(data_save)
    with open(data_save, 'wb') as fs:
        pickle.dump(title_dict, fs)
    print('Done')


def main():
    dataset_name = "Arts"
    path_meta = '/data/' + dataset_name + '/meta_' + amazon_dataset2fullname[dataset_name] + '.json'
    data_save = '/data/' + dataset_name + '/stmap.pkl'
    smap_tmap_extract(path_meta, data_save)


if __name__ == '__main__':
    main()