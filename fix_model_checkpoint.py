#!/usr/bin/env python3
"""
Fix Model Checkpoint - Download missing configuration files
"""

import os
import shutil
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_model_checkpoint(checkpoint_path="models/results/checkpoint-2142"):
    """Fix the model checkpoint by downloading missing configuration files"""
    print("🔧 Fixing Model Checkpoint...")
    print("="*50)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path not found: {checkpoint_path}")
        return False
    
    try:
        # Check what files are missing
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if not missing_files:
            print("✅ All required files are present!")
            return True
        
        print(f"Missing files: {missing_files}")
        print("Downloading from base model...")
        
        # Download configuration from base model
        base_model_name = "nlpaueb/legal-bert-base-uncased"
        
        # Download config
        print("Downloading config.json...")
        config = AutoConfig.from_pretrained(base_model_name)
        config.save_pretrained(checkpoint_path)
        
        # Download tokenizer files
        print("Downloading tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Verify files were downloaded
        print("Verifying downloaded files...")
        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"✅ {file}: {size:.1f} KB")
            else:
                print(f"❌ {file}: Not found")
        
        print("✅ Model checkpoint fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing checkpoint: {str(e)}")
        return False

def test_fixed_model():
    """Test if the fixed model can be loaded"""
    print("\n🧪 Testing Fixed Model...")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        checkpoint_path = "models/results/checkpoint-2142"
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        print("✅ Model loaded successfully")
        
        # Check model info
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Number of labels: {model.config.num_labels}")
        print(f"Labels: {list(model.config.id2label.values())}")
        
        # Test inference
        print("Testing inference...")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Simple test
        test_text = "This agreement is between ABC Corporation and XYZ Ltd."
        inputs = tokenizer(test_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        print("✅ Inference test successful")
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed model: {str(e)}")
        return False

def main():
    """Main function to fix model checkpoint"""
    print("🔧 Legal Document Analyzer - Fix Model Checkpoint")
    print("="*60)
    
    # Fix the checkpoint
    if fix_model_checkpoint():
        print("\n✅ Checkpoint fixed successfully!")
        
        # Test the fixed model
        if test_fixed_model():
            print("\n🎉 Model is now ready to use!")
            print("\nNext steps:")
            print("1. Run: python check_model_status.py")
            print("2. Run: python test_trained_model.py")
        else:
            print("\n❌ Model testing failed. Please check the error messages.")
    else:
        print("\n❌ Failed to fix checkpoint. Please check the error messages.")

if __name__ == "__main__":
    main() 