class Vocabulary:
    def __init__(self):
        #Initialize special tokens
        # <pad> token is used to make all the sequences the same length
        # <unk> token is used for characters we havent seen in training

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        #Create twp dictionaries:
        #Char2idx : maps characters to unique indices (for model input)
        #idxtochar : maps indices back to characters (for human interpretation)

        self.char2idx = {
            self.pad_token: 0,#Always assing 0 to  pad token
            self.unk_token : 1, #Always assing 1 to unknown token
        }

        self.idx2char = {
            0 : self.pad_token,
            1: self.unk_token
        }
        #Keeping track of next available index for new characters
        self.next_idx = 2 #starting from 2 since 0 and 1 are sepcial tokens
    
    def lookup_index(self,char):
        """
        Convert a character to its corresponding index
        Args:
        char: the character to look up
        Returns:
         The index for this character or the <unk> index if not found

        """
        return self.char2idx.get(char, self.char2idx[self.unk_token])
    
    def lookup_char(self,idx):
        """
        Convert an index back to its correspoinding character
        Args:
          idx: the index to lookup
        Returns:
           The character for this index, or <unk> if not found
        """
        return self.idx2char.get(idx, self.unk_token)
    def __len__(self):
        """
        Get the size of the vocabulary
        Returns:
          Total number of unique characters (including special token)
        """
        return len(self.char2idx)
    