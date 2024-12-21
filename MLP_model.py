import torch 
import torch.nn as nn


class SurnameMLP(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim,num_naionalities):
        """
        Initialize the MLP model for surname classification
        Args:
          vocab_size : Number of unique character in the vocabulary
          embedding_dim : Size of the Character Embeddings
          hidden_dim : Size of the hidden layer
          num_nationalities : Number of nationality classes to predict
        """
        super.__init__()
        #Embedding layer: conversts character indices to dense vectors
        #This helps the model learn relationships between characters
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size, #Size of vocabulary
            embedding_dim = embedding_dim, #Size of each embedding
            padding_idx = 0, #Index of padding token (will be Zeroed)
        )

        #First Fully Connected layers
        #Input size is embedding_dim * 20 beacuse:
        # - Each characters becomes a vector of size embedding_dim
        # - We have 20 characters (max_surname_length)
        # - We flatten all these vectors into one

        self.fc1 = nn.Linear(embedding_dim * 20, hidden_dim)

        #Relu activation function: introduces non-linearity
        self.relu = nn.ReLU()

        #Dropout layer: helps prevent overfitting
        #Randomly zeroes 30% of the elements during training
        self.dropout = nn.Dropout(0.3)

        #Output layer: prodcues logits for each nationality
        self.fc2 = nn.Linear(hidden_dim, num_naionalities)

    def forward(self,x):
        """
        Forward pass of the model
        Args: 
           x: Tensor of shape (batch_size, max_surname_length) containing character indices
        Returns: Logits for each nationality class
        """
        #Convert character indices to embeddings
        #Shape : (batch_size, max_surname_lenght, embedding_dim)
        embedded = self.embedding(x)

        #Flatten the embedding to feed into linear layer
        #Shape: (batch_size, max_surname_length * embedding_dim)
        flattened = embedded.view(embedded.size(0),-1)

        #Pass through first layer and apply ReLU activation
        hidden = self.fc1(flattened)
        hidden = self.relu(hidden)

        #Apply dropout for regularization
        hidden = self.dropout(hidden)

        #Get final logits
        # Shape : (batch_size, num_nationalities)
        logits = self.fc2(hidden)

        return logits 