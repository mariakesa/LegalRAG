
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
import random

# Initialize the data collator
tokenizer = AutoTokenizer.from_pretrained('joelniklaus/legal-xlm-roberta-base')
data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

class ContrastiveDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset
        self.num_documents = len(self.dataset)
    
    def __len__(self):
        # Define the number of samples per epoch
        return self.num_documents * 10  # Adjust as needed
    
    def __getitem__(self, idx):
        # Pick a document at random
        doc_idx = random.randint(0, self.num_documents - 1)
        document = self.dataset[doc_idx]
        tokens = document['tokens']
        num_tokens = len(tokens)
        
        if num_tokens < 2:
            # Handle short documents
            anchor_tokens = tokens
            positive_tokens = tokens
        elif num_tokens <= 512:
            # Split into two halves
            mid_point = num_tokens // 2
            anchor_tokens = tokens[:mid_point]
            positive_tokens = tokens[mid_point:]
        else:
            # Sample a random starting index
            max_start = num_tokens - 2 * 512
            if max_start > 0:
                start_idx = random.randint(0, max_start)
            else:
                start_idx = 0
            # Extract spans
            anchor_tokens = tokens[start_idx : start_idx + 512]
            positive_tokens = tokens[start_idx + 512 : start_idx + 2 * 512]
        
        # Prepare the inputs as dictionaries
        anchor = {'input_ids': anchor_tokens}
        positive = {'input_ids': positive_tokens}
        
        return {'anchor': anchor, 'positive': positive}


def contrastive_data_collator(features):
    anchors = [feature['anchor'] for feature in features]
    positives = [feature['positive'] for feature in features]
    
    # Use DataCollatorWithPadding to pad anchors and positives separately
    batch_anchors = data_collator(anchors)
    batch_positives = data_collator(positives)
    
    return batch_anchors, batch_positives