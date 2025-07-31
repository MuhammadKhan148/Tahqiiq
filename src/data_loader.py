import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset, load_from_disk
from typing import Dict, Optional, List
from src.utils import load_cuad_dataset, get_data_collator, prepare_cuad_dataset, compute_metrics

class LegalBertDataset(Dataset):
    """PyTorch Dataset for LEGAL-BERT tokenized data"""
    
    def __init__(self, split_dir: str, manifest_path: str):
        """Initialize dataset from split directory and manifest"""
        self.split_dir = split_dir
        self.manifest_path = manifest_path
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        
        # Load tensors
        self.input_ids = torch.load(os.path.join(split_dir, 'input_ids.pt'))
        self.attention_masks = torch.load(os.path.join(split_dir, 'attention_masks.pt'))
        self.labels = torch.load(os.path.join(split_dir, 'labels.pt'))
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.samples[idx]
        tensor_idx = sample['tensor_index']
        
        return {
            'input_ids': self.input_ids[tensor_idx],
            'attention_mask': self.attention_masks[tensor_idx],
            'labels': self.labels[tensor_idx],
            'id': sample['id'],
            'original_index': sample['original_index']
        }

class LegalDataLoaderFactory:
    """Factory class to create data loaders for LEGAL-BERT splits"""
    
    def __init__(self, splits_dir: str = "data/splits"):
        """Initialize the factory with splits directory"""
        self.splits_dir = splits_dir
        self.datasets = {}
        self._load_splits_metadata()
    
    def _load_splits_metadata(self):
        """Load splits metadata"""
        metadata_path = os.path.join(self.splits_dir, 'splits_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded splits metadata: {self.metadata['total_samples']} total samples")
        for split_name, split_info in self.metadata['splits'].items():
            print(f"   {split_name.upper()}: {split_info['size']} samples")
    
    def get_dataset(self, split_name: str) -> LegalBertDataset:
        """Get dataset for a specific split"""
        if split_name not in self.datasets:
            split_dir = os.path.join(self.splits_dir, split_name)
            manifest_path = os.path.join(self.splits_dir, f"{split_name}_manifest.jsonl")
            
            if not os.path.exists(split_dir) or not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Split '{split_name}' not found in {self.splits_dir}")
            
            self.datasets[split_name] = LegalBertDataset(split_dir, manifest_path)
        
        return self.datasets[split_name]
    
    def get_dataloader(self, split_name: str, batch_size: int = 8, shuffle: Optional[bool] = None) -> DataLoader:
        """Get PyTorch DataLoader for a specific split"""
        dataset = self.get_dataset(split_name)
        
        # Default shuffle behavior
        if shuffle is None:
            shuffle = (split_name == 'train')
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues on Windows
            pin_memory=torch.cuda.is_available()
        )
    
    def get_all_dataloaders(self, batch_size: int = 8) -> Dict[str, DataLoader]:
        """Get all data loaders (train, val, test)"""
        loaders = {}
        for split_name in ['train', 'val', 'test']:
            try:
                loaders[split_name] = self.get_dataloader(split_name, batch_size)
            except FileNotFoundError:
                print(f"Error loading {split_name.upper()}: FileNotFoundError")
        
        return loaders
    
    def prepare_cuad_dataset(self, save_path: str = "./data/tokenized_cuad") -> Dict[str, HFDataset]:
        """Prepare CUAD dataset for token classification training"""
        print("Preparing CUAD dataset for token classification...")
        
        # Load dataset and labels
        dataset, label_list, num_labels, label2id, id2label = load_cuad_dataset()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        # Prepare data using enhanced function
        processed_data = prepare_cuad_dataset(tokenizer, dataset, label2id)
        
        # Create Dataset objects
        tokenized_datasets = {
            "train": HFDataset.from_list(processed_data["train"]),
            "validation": HFDataset.from_list(processed_data["validation"])
        }
        
        # Save to disk
        os.makedirs(save_path, exist_ok=True)
        tokenized_datasets["train"].save_to_disk(os.path.join(save_path, "train"))
        tokenized_datasets["validation"].save_to_disk(os.path.join(save_path, "validation"))
        
        print(f"Dataset prepared and saved to {save_path}")
        print(f"Train samples: {len(tokenized_datasets['train'])}")
        print(f"Validation samples: {len(tokenized_datasets['validation'])}")
        
        return tokenized_datasets
    
    def load_cuad_datasets(self, data_path: str = "./data/tokenized_cuad") -> Dict[str, HFDataset]:
        """Load prepared CUAD datasets from disk"""
        print(f"Loading prepared CUAD datasets from {data_path}...")
        
        tokenized_datasets = {
            "train": load_from_disk(os.path.join(data_path, "train")),
            "validation": load_from_disk(os.path.join(data_path, "validation"))
        }
        
        print(f"Loaded {len(tokenized_datasets['train'])} train samples")
        print(f"Loaded {len(tokenized_datasets['validation'])} validation samples")
        
        return tokenized_datasets
    
    def train_cuad_model(self, tokenized_datasets: Dict[str, HFDataset], 
                         model_save_path: str = "./models/fine_tuned_legalbert_cuad") -> Trainer:
        """Train CUAD token classification model"""
        print("Training CUAD token classification model...")
        
        # Load labels
        _, label_list, num_labels, label2id, id2label = load_cuad_dataset()
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Training args
        training_args = TrainingArguments(
            output_dir="./models/results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced for memory
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Tokenizer for collator
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        data_collator = get_data_collator(tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, id2label),
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Model trained and saved to {model_save_path}")
        
        return trainer
    
    def get_split_info(self) -> Dict:
        """Get information about all splits"""
        return self.metadata
    
    def print_sample(self, split_name: str = 'train', sample_idx: int = 0):
        """Print a sample from the dataset for debugging"""
        dataset = self.get_dataset(split_name)
        sample = dataset[sample_idx]
        
        print(f"\nSample from {split_name} split (index {sample_idx}):")
        print(f"   ID: {sample['id']}")
        print(f"   Original index: {sample['original_index']}")
        print(f"   Label: {sample['labels'].item()}")
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Attention mask shape: {sample['attention_mask'].shape}")
        print(f"   First 10 tokens: {sample['input_ids'][:10].tolist()}")

def test_data_loaders():
    """Test function to validate data loaders work correctly"""
    print("\nTesting data loaders...")
    
    # Initialize factory
    factory = LegalDataLoaderFactory()
    
    # Test individual datasets
    for split_name in ['train', 'val', 'test']:
        try:
            dataset = factory.get_dataset(split_name)
            print(f"{split_name.upper()} dataset: {len(dataset)} samples")
            
            # Test getting a sample
            if len(dataset) > 0:
                sample = dataset[0]
                assert 'input_ids' in sample
                assert 'attention_mask' in sample
                assert 'labels' in sample
                print(f"   Sample 0 - Label: {sample['labels'].item()}, Shape: {sample['input_ids'].shape}")
        
        except FileNotFoundError as e:
            print(f"Error in {split_name.upper()}: {e}")
    
    # Test data loaders
    print("\nTesting data loaders...")
    loaders = factory.get_all_dataloaders(batch_size=2)
    
    for split_name, loader in loaders.items():
        print(f"{split_name.upper()} loader: {len(loader)} batches")
        
        # Test one batch
        for batch in loader:
            print(f"   Batch shape - Input IDs: {batch['input_ids'].shape}, Labels: {batch['labels'].shape}")
            break
    
    # Print a sample
    factory.print_sample('train', 0)
    
    print("\nAll data loader tests passed!")

def main():
    """Main function to demonstrate data loader usage"""
    print("LEGAL-BERT Data Loader Interface - Muhammad Abdullah Khan")
    
    # Test the data loaders
    test_data_loaders()
    
    # Show usage example
    print("\nUsage Example:")
    print("```python")
    print("from src.data_loader import LegalDataLoaderFactory")
    print("")
    print("# Initialize factory")
    print("factory = LegalDataLoaderFactory()")
    print("")
    print("# Get individual dataset")
    print("train_dataset = factory.get_dataset('train')")
    print("")
    print("# Get data loader")
    print("train_loader = factory.get_dataloader('train', batch_size=8)")
    print("")
    print("# Get all loaders")
    print("loaders = factory.get_all_dataloaders(batch_size=16)")
    print("```")

if __name__ == "__main__":
    main() 