import torch
from torch.utils.data import Dataset
import pandas as pd


class SurnameDataset(Dataset):
    def __init__(self, csv_file, vocab,nationality2isx, max_surname_length = 20):
        """
        Initialize the dataset
        Args:
            csv_file: Path to the csv containiing surnames and nationalities
            vocab: Vocabulary object for characters-to-index conversion
            nationality2idx: Dictionary mapping nationality to index
            max_surname_lenght : Maximum length for padding/truncating surnames
        """
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.nationality2idx = nationality2isx
        self.max_surname_length = max_surname_length

    def _build_vocab(self):
        """
        Process all the surnames to build the vocabulary
        This ensure we have indices for all the characters that appear in the data
        """
        for surname in self.df['surname']:
            #Add each character in each surnameto the vocabulary
            for char in surname:
                self.vocab.add_character(char)
    def _vectorize_surname(self,surname):
        """
        Convert a surname string into a vector of character indices
        Args:
           surname: The surname string to convert
        Returns:
           torch.tensor containing characters indices, padded/truncated to max_length
        """
        indices = [self.vocab.lookup_index(char) for char in surname]

        #Handle length
        if len(indices) < self.max_surname_length:
            #if surname is too short, pad with pad_token
            indices += [self.vocab.char2idx[self.vocab.pad_token]] * (self.max_surname_length - len(indices))
        else:
            #If surname is too long, truncate
            indices = indices[:self.max_surname_length]

    def __len__(self):
        """
        Get the total number of surnames in the dataset
        Required by Pytorch's Dataset class
        """
        return len(self.df)
    
    def __getitem___(self,idx):
        """
        Get a single surname and it's nationality 
        Required by PyTorch's Dataset class
        Args:
            idx: The index of the item to get
        Returns:
            Dictionary containing the vectorized surname and nationality index
        """
        surname = self.df.iloc[idx]['surname']
        nationality = self.df.iloc['nationality']

        #Convert surname to vector of character indices
        surname_vector = self._vectorize_surname(surname)

        #convert nationality to index
        nationality_idx = self.nationality2idx[nationality]

        return {
            'surname': surname_vector,
            'nationality': torch.tensor(nationality_idx, dtype=torch.long)
        }