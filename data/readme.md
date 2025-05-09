# Readme

The datasets in our experiments are derived from two primary sources: (1) the raw meta information (e.g., title, review) downloaded from [Amazon review](https://nijianmo.github.io/amazon/index.html) and (2) the preprocessed interactions (i.e., item sequences) obtained from [UniSRec](https://github.com/RUCAIBox/UniSRec). 
Please preprocess the reviews and records based on the scripts. Let's take the Office dataset as an example, the preprocessed dataset should be:

```
Office
├─title_review_summary_descroption
├──test.pkl
├──train.pkl
├──val.pkl
├─negative_title
├──user_item_negitem_nge_title_seq_test.pkl
├──user_item_negitem_nge_title_seq_train.pkl
├──user_item_negitem_nge_title_seq_val.pkl
├─Office_products_5.json
├─meta_Office_Products.json
├─Office_products_5.json
└stmap.pkl
```

Or you can download the processed datasets from [here]().
