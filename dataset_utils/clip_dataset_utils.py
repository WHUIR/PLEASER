from torch.utils.data import Dataset
import pickle


def add_datasets_args(parent_args):
    parser = parent_args.add_argument_group('clip rec2seq data args')
    parser.add_argument(
        "--datasets_path_train", type=str, default=None, required=True, nargs='+',
        help="A folder containing the training data of instance text.",
    )
    parser.add_argument(
        "--datasets_path_val", type=str, default=None, required=True, nargs='+',
        help="A folder containing the val data of instance text.",
    )
    parser.add_argument(
        "--datasets_path_test", type=str, default=None, required=True, nargs='+',
        help="A folder containing the test data of instance text.",
    )
    parser.add_argument("--thres", type=float, default=0.2)
    return parent_args


class PKLDataset(Dataset):
    def __init__(self, data_raw):
        self.data_raw = data_raw

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        data_temp = self.data_raw[idx]        
        return data_temp


def process_pool_read_pkl_dataset_raw(input_path):
    with open(input_path[0], 'rb') as f:
        data_raw = pickle.load(f)
    pkl_dataset = PKLDataset(data_raw) 
    return pkl_dataset


def load_data_raw(args):
    data_train = process_pool_read_pkl_dataset_raw(args.datasets_path_train)
    data_val = process_pool_read_pkl_dataset_raw(args.datasets_path_val)
    data_test = process_pool_read_pkl_dataset_raw(args.datasets_path_test)
    return {'train': data_train, 'val': data_val, 'test': data_test}
     

def main():
    pass


if __name__ == "__main__":
    main()
