import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmModel

import argparse
from pathlib import Path
import random


# from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
import torch
from transformers import EsmTokenizer
from transformers.utils import logging

from esmtherm.util import write_json, read_json

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

from torch.utils.data import Dataset, DataLoader, random_split


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, sequence_column='sequence', target_column='target'):
        """
        Args:
            dataframe (pd.DataFrame): input dataframe
            sequence_column (str): column in the dataframe that contains the sequences
            target_column (str): column in the dataframe that contains the targets
        """
        self.sequences = dataframe[sequence_column].tolist()
        # self.id = dataframe['id'].tolist()
        # self.targets = dataframe[target_column].tolist()
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx] #, self.targets[idx]  int(self.id[idx]),  # Returns the sequence and target at the given index


# def create_data_loader(sequences, batch_size=32):
#     """ Creates a DataLoader for batching sequences. """
#     dataset = ProteinDataset(sequences)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=False)
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='./kaggle_op/',  help='Dataset directory')
parser.add_argument('--csv', type=str, default='../test.csv')
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=25)
# parser.add_argument('--split_csv', type=str, default=None,
args = parser.parse_args()

random.seed(args.seed)


def main():
    # Initialize tokenizer and model from Hugging Face Transformers
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model.eval()  # Set the model to evaluation mode

    dataset_dir = Path(args.output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


    dataset = DataFrameDataset(pd.read_csv(args.csv))
    # split into training and test data 
    #train_size = int((1-2*args.test_split) * len(dataset))
    #test_size = len(dataset) - train_size
    train_dataset,_ = random_split(dataset, [len(dataset), 0])

    print(len(train_dataset))

    # Create data loaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


    # all_sequences = []
    # all_ids = []
    # for batch in train_loader:
    #     print(batch)
    #     id = [int(entry) for entry in batch[0]]
    #     sequences = batch[1]
    #     # zip(*batch)  # This assumes that targets exist, modify if not
    #     all_sequences.extend(sequences)
    #     all_ids.extend(id)

    # Save as DataFrame
    # df = pd.DataFrame({
    #     'id': all_ids,
    #     'sequence': all_sequences,
    #     # Uncomment the next line if targets are used
    #     # 'target': all_targets  
    # })

    # # Optionally, save to a CSV file
    # df.to_csv('./kaggle_op/test/all_testing.csv', index=False)



    # print(train_loader)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # torch.save(train_loader,f'{args.output_dir}/test/all_output.pt')
    dataset_dir_train = Path(args.output_dir + '/test/')
    dataset_dir_train.mkdir(parents=True, exist_ok=True)

    for i, (batch_sequences) in enumerate(train_loader):
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.detach()
        pooled_output = embeddings.mean(dim=1)
        torch.save((pooled_output), f'{args.output_dir}/test/test_embeddings_targets_batch_{i}.pt')




if __name__ == "__main__":
    main()