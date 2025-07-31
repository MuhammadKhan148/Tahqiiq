#!/usr/bin/env python3
"""
Simple script to train the CUAD model
"""

import sys
sys.path.append('.')

def train_model():
    """Train the CUAD model"""
    print("Training CUAD Model...")
    
    try:
        from src.data_loader import LegalDataLoaderFactory
        
        factory = LegalDataLoaderFactory()
        tokenized_datasets = factory.load_cuad_datasets()
        
        print("Training model (this may take a while)...")
        trainer = factory.train_cuad_model(tokenized_datasets)
        
        print("✓ Model trained successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_model() 