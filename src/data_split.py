import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class LegalDataSplitter:
    def __init__(self, tokenized_dir="data/tokenized", splits_dir="data/splits"):
        """Initialize data splitter for LEGAL-BERT tokenized data"""
        self.tokenized_dir = tokenized_dir
        self.splits_dir = splits_dir
        self.seed = 42
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        print("Initializing LegalDataSplitter...")
        print(f"Tokenized data: {tokenized_dir}")
        print(f"Splits output: {splits_dir}")
    
    def load_tokenized_data(self):
        """Load tokenized tensors and metadata"""
        print("Loading tokenized data...")
        
        # Load tensors
        input_ids = torch.load(os.path.join(self.tokenized_dir, "input_ids.pt"))
        attention_masks = torch.load(os.path.join(self.tokenized_dir, "attention_masks.pt"))
        labels = torch.load(os.path.join(self.tokenized_dir, "labels.pt"))
        
        # Load metadata
        with open(os.path.join(self.tokenized_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(input_ids)} samples")
        print(f"Label distribution: {metadata['label_distribution']}")
        
        return input_ids, attention_masks, labels, metadata
    
    def create_splits(self, input_ids, attention_masks, labels, train_size=0.8, val_size=0.1, test_size=0.1):
        """Split data into train/val/test with simple random splitting for small datasets"""
        print(f"Creating splits: train={train_size:.0%}, val={val_size:.0%}, test={test_size:.0%}")
        
        # Verify split ratios sum to 1.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Convert labels to numpy
        labels_np = labels.numpy()
        indices = np.arange(len(labels_np))
        total_samples = len(labels_np)
        
        print(f"Dataset: {total_samples} samples")
        print("Using simple random splitting for small dataset")
        
        # Calculate actual split sizes
        test_samples = max(1, int(total_samples * test_size))
        val_samples = max(1, int(total_samples * val_size))
        train_samples = total_samples - test_samples - val_samples
        
        if train_samples < 1:
            train_samples = 1
            val_samples = min(val_samples, total_samples - train_samples - test_samples)
            test_samples = total_samples - train_samples - val_samples
        
        # Shuffle and split
        shuffled_indices = np.random.permutation(indices)
        train_indices = shuffled_indices[:train_samples]
        val_indices = shuffled_indices[train_samples:train_samples+val_samples]
        test_indices = shuffled_indices[train_samples+val_samples:]
        
        # Create split dictionaries
        splits = {
            'train': {
                'indices': train_indices,
                'input_ids': input_ids[train_indices],
                'attention_masks': attention_masks[train_indices],
                'labels': labels[train_indices]
            },
            'val': {
                'indices': val_indices,
                'input_ids': input_ids[val_indices],
                'attention_masks': attention_masks[val_indices],
                'labels': labels[val_indices]
            },
            'test': {
                'indices': test_indices,
                'input_ids': input_ids[test_indices],
                'attention_masks': attention_masks[test_indices],
                'labels': labels[test_indices]
            }
        }
        
        # Print split statistics
        for split_name, split_data in splits.items():
            label_counts = Counter(split_data['labels'].numpy())
            print(f"{split_name.upper()}: {len(split_data['indices'])} samples, labels: {dict(label_counts)}")
        
        return splits
    
    def save_split_tensors(self, splits):
        """Save tensor data for each split"""
        print("Saving split tensors...")
        
        os.makedirs(self.splits_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            split_dir = os.path.join(self.splits_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Save tensors
            torch.save(split_data['input_ids'], os.path.join(split_dir, 'input_ids.pt'))
            torch.save(split_data['attention_masks'], os.path.join(split_dir, 'attention_masks.pt'))
            torch.save(split_data['labels'], os.path.join(split_dir, 'labels.pt'))
            
            print(f"Saved {split_name} tensors to {split_dir}")
    
    def create_manifests(self, splits, metadata):
        """Create manifest files for each split"""
        print("Creating manifest files...")
        
        for split_name, split_data in splits.items():
            manifest_path = os.path.join(self.splits_dir, f"{split_name}_manifest.jsonl")
            
            with open(manifest_path, 'w') as f:
                for i, (idx, label) in enumerate(zip(split_data['indices'], split_data['labels'])):
                    manifest_entry = {
                        'id': f"{split_name}_{i:06d}",
                        'original_index': int(idx),
                        'label': int(label),
                        'tensor_path': f"{split_name}/",
                        'tensor_index': i
                    }
                    f.write(json.dumps(manifest_entry) + '\n')
            
            print(f"Created manifest: {manifest_path}")
    
    def save_split_metadata(self, splits, original_metadata):
        """Save metadata for splits"""
        split_metadata = {
            'seed': self.seed,
            'total_samples': sum(len(split['indices']) for split in splits.values()),
            'splits': {
                name: {
                    'size': len(split['indices']),
                    'label_distribution': {str(k): int(v) for k, v in Counter(split['labels'].numpy()).items()}
                }
                for name, split in splits.items()
            },
            'original_metadata': original_metadata
        }
        
        metadata_path = os.path.join(self.splits_dir, 'splits_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        print(f"Saved split metadata: {metadata_path}")
        return split_metadata
    
    def validate_splits(self, splits):
        """Validate that splits have no overlap and correct properties"""
        print("Validating splits...")
        
        all_indices = set()
        total_samples = 0
        
        for split_name, split_data in splits.items():
            indices = set(split_data['indices'])
            
            # Check for overlap
            overlap = all_indices.intersection(indices)
            assert len(overlap) == 0, f"Found overlap between splits: {overlap}"
            
            all_indices.update(indices)
            total_samples += len(indices)
            
            # Validate tensor shapes
            n_samples = len(split_data['indices'])
            assert split_data['input_ids'].shape[0] == n_samples
            assert split_data['attention_masks'].shape[0] == n_samples
            assert split_data['labels'].shape[0] == n_samples
            assert split_data['input_ids'].shape[1] == 512  # LEGAL-BERT max length
        
        print(f"Validation passed: {total_samples} total samples, no overlaps")
        return True

def main():
    """Main function to create data splits"""
    print("Starting data splitting for Muhammad Abdullah Khan...")
    
    # Initialize splitter
    splitter = LegalDataSplitter()
    
    # Load tokenized data
    input_ids, attention_masks, labels, metadata = splitter.load_tokenized_data()
    
    # Create splits
    splits = splitter.create_splits(input_ids, attention_masks, labels)
    
    # Validate splits
    splitter.validate_splits(splits)
    
    # Save split tensors
    splitter.save_split_tensors(splits)
    
    # Create manifests
    splitter.create_manifests(splits, metadata)
    
    # Save metadata
    split_metadata = splitter.save_split_metadata(splits, metadata)
    
    print("\nData splitting complete!")
    print(f"Final split sizes:")
    for split_name, split_info in split_metadata['splits'].items():
        print(f"   {split_name.upper()}: {split_info['size']} samples")
    
    return splits, split_metadata

if __name__ == "__main__":
    main() 