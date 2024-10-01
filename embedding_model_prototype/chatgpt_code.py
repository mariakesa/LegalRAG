# Import necessary libraries
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# Step 1: Load and Tokenize the Dataset
from datasets import load_dataset

# Load the dataset
#corpus = load_dataset('umarbutler/open-australian-legal-corpus', split='corpus', keep_in_memory=False)

# Load the dataset
corpus = load_dataset(
    'umarbutler/open-australian-legal-corpus',
    split='corpus',
    keep_in_memory=False
)

# Shuffle the dataset to ensure random selection
corpus = corpus.shuffle(seed=42)

# Select the first 20,000 documents
corpus = corpus.select(range(20000))

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('joelniklaus/legal-xlm-roberta-base')

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Assign an appropriate token if missing

# Define a function to tokenize the text and compute token counts
def tokenize_function(examples):
    # Tokenize the text without truncation or padding
    tokenized = tokenizer(examples['text'], truncation=False, padding=False)
    # Compute the number of tokens for each example
    num_tokens = [len(input_ids) for input_ids in tokenized['input_ids']]
    # Add the number of tokens and tokens to the examples
    examples['num_tokens'] = num_tokens
    examples['tokens'] = tokenized['input_ids']
    return examples

# Apply the tokenize_function to the dataset
corpus = corpus.map(tokenize_function, batched=True, batch_size=1000)


# Step 2: Define the Custom Dataset Class
class ContrastiveDataset(Dataset):
    def __init__(self, tokenized_dataset, multiplier=1):
        self.dataset = tokenized_dataset
        self.num_documents = len(self.dataset)
        self.multiplier = multiplier  # Determines how many samples per document per epoch
    
    def __len__(self):
        # Define the number of samples per epoch
        return self.num_documents * self.multiplier  # Adjust the multiplier as needed
    
    def __getitem__(self, idx):
        # Pick a document at random
        doc_idx = random.randint(0, self.num_documents - 1)
        document = self.dataset[doc_idx]
        tokens = document['tokens']
        num_tokens = len(tokens)
        
        if num_tokens <=  2 * 512:
            # Handle short documents by splitting into two halves
            mid_point = num_tokens // 2
            anchor_tokens = tokens[:mid_point]
            positive_tokens = tokens[mid_point:]
        else:
            # Sample a random starting index for longer documents
            max_start = num_tokens - 2 * 512
            if max_start > 0:
                start_idx = random.randint(0, max_start)
            else:
                start_idx = 0
            # Extract two spans of 512 tokens each
            anchor_tokens = tokens[start_idx : start_idx + 512]
            positive_tokens = tokens[start_idx + 512 : start_idx + 2 * 512]
        
        # Prepare the inputs as dictionaries
        anchor = {'input_ids': anchor_tokens}
        positive = {'input_ids': positive_tokens}
        
        return {'anchor': anchor, 'positive': positive}

# Step 3: Define the Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Computes the contrastive loss between two sets of embeddings.
        
        Args:
            z_i: Tensor of shape (batch_size, embedding_dim)
            z_j: Tensor of shape (batch_size, embedding_dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)
        # Concatenate the embeddings
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, embedding_dim)
        # Compute cosine similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature  # Shape: (2 * batch_size, 2 * batch_size)
        # Fill the diagonal with -inf to exclude self-similarity
        sim_matrix.fill_diagonal_(-float('inf'))
        # Create labels
        labels = torch.arange(batch_size).to(z.device)
        # Compute loss for anchors
        loss_i = F.cross_entropy(sim_matrix[:batch_size, :], labels)
        # Compute loss for positives
        loss_j = F.cross_entropy(sim_matrix[batch_size:, :], labels)
        # Average the loss
        loss = (loss_i + loss_j) / 2
        return loss

# Step 4: Create the DataLoader with Hugging Face's DataCollatorWithPadding
# Initialize the data collator
data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

def contrastive_data_collator_fn(features):
    """
    Custom collate function for contrastive learning.
    
    Args:
        features: List of dictionaries with 'anchor' and 'positive' keys.
    
    Returns:
        Tuple of batched anchors and positives.
    """
    anchors = [feature['anchor'] for feature in features]
    positives = [feature['positive'] for feature in features]
    
    # Use DataCollatorWithPadding to pad anchors and positives separately
    batch_anchors = data_collator(anchors)
    batch_positives = data_collator(positives)
    
    return batch_anchors, batch_positives

# Instantiate the ContrastiveDataset
contrastive_dataset = ContrastiveDataset(corpus, multiplier=1)  # Adjust multiplier as needed

# Create the DataLoader
data_loader = DataLoader(
    contrastive_dataset,
    batch_size=5,            # Adjust based on your GPU memory
    shuffle=True,             # Shuffle the data
    num_workers=4,            # Adjust based on your CPU cores
    collate_fn=contrastive_data_collator_fn,
    pin_memory=True           # Improves performance if using GPU
)

# Step 5: Define the Training Function
def train_contrastive(model, dataloader, loss_fn, optimizer, device, epochs=3):
    """
    Trains the model using contrastive learning.
    
    Args:
        model: The transformer model to train.
        dataloader: DataLoader providing batches of anchor-positive pairs.
        loss_fn: Contrastive loss function.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the training on (CPU or GPU).
        epochs: Number of training epochs.
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, (batch_anchors, batch_positives) in enumerate(dataloader):
            # Move batch to device
            batch_anchors = {k: v.to(device) for k, v in batch_anchors.items()}
            batch_positives = {k: v.to(device) for k, v in batch_positives.items()}
            
            # Forward pass for anchors
            anchor_outputs = model(**batch_anchors)
            # Forward pass for positives
            positive_outputs = model(**batch_positives)
            
            # Extract embeddings (using [CLS] token)
            # For RoBERTa-based models, the first token ([CLS] or <s>) is typically used
            z_i = anchor_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            z_j = positive_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            
            # Normalize the embeddings
            z_i = F.normalize(z_i, p=2, dim=1)
            z_j = F.normalize(z_j, p=2, dim=1)
            
            # Compute the contrastive loss
            loss = loss_fn(z_i, z_j)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'==> Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}')
        
        # Optionally, save the model checkpoint after each epoch
        # torch.save(model.state_dict(), f'contrastive_model_epoch_{epoch+1}.pth')

# Step 6: Initialize the Model, Loss Function, and Optimizer
# Load the pre-trained model
model = AutoModel.from_pretrained('joelniklaus/legal-xlm-roberta-base')

# Initialize the contrastive loss function
contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Step 7: Set the Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 8: Start Training
# Define the number of epochs
num_epochs = 1  # Adjust as needed

import time

start=time.time()
# Train the model
train_contrastive(model, data_loader, contrastive_loss_fn, optimizer, device, epochs=num_epochs)
end=time.time()

print(f"Time taken to train the model: {end-start} seconds")

# Step 9: Save the Trained Model (Optional)
torch.save(model.state_dict(), 'trained_contrastive_model.pth')
