#!/usr/bin/env python3
"""
Check Model Status - Verify trained model and training logs
"""

import os
import json
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_files():
    """Check if model files exist and are accessible"""
    print("🔍 Checking Model Files...")
    print("="*50)
    
    model_path = "models/results/checkpoint-2142"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    # Check for model files
    model_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    found_files = []
    missing_files = []
    
    for file in model_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            found_files.append((file, size))
        else:
            missing_files.append(file)
    
    print(f"✅ Found {len(found_files)} model files:")
    for file, size in found_files:
        print(f"   - {file}: {size:.1f} MB")
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    
    return len(found_files) > 0

def check_model_config():
    """Check model configuration"""
    print("\n🔧 Checking Model Configuration...")
    print("="*50)
    
    try:
        from transformers import AutoConfig
        
        config_path = "models/results/checkpoint-2142/config.json"
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained("models/results/checkpoint-2142")
            
            print(f"✅ Model configuration loaded successfully")
            print(f"   - Model type: {config.model_type}")
            print(f"   - Number of labels: {config.num_labels}")
            print(f"   - Hidden size: {config.hidden_size}")
            print(f"   - Number of layers: {config.num_hidden_layers}")
            print(f"   - Attention heads: {config.num_attention_heads}")
            
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return False

def check_training_logs():
    """Check training logs"""
    print("\n📊 Checking Training Logs...")
    print("="*50)
    
    logs_path = "logs"
    if not os.path.exists(logs_path):
        print(f"❌ Logs directory not found: {logs_path}")
        return False
    
    log_files = [f for f in os.listdir(logs_path) if f.endswith('.tfevents')]
    
    if not log_files:
        print("❌ No training log files found")
        return False
    
    print(f"✅ Found {len(log_files)} training log files:")
    for log_file in log_files:
        size = os.path.getsize(os.path.join(logs_path, log_file)) / 1024  # KB
        print(f"   - {log_file}: {size:.1f} KB")
    
    return True

def check_device_compatibility():
    """Check device compatibility"""
    print("\n💻 Checking Device Compatibility...")
    print("="*50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        print(f"Memory allocated: {memory_allocated:.2f} GB")
        print(f"Memory reserved: {memory_reserved:.2f} GB")
    
    # Check CPU
    print(f"CPU cores: {os.cpu_count()}")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking Dependencies...")
    print("="*50)
    
    required_packages = [
        "torch",
        "transformers", 
        "pdfplumber",
        "pandas",
        "numpy",
        "seqeval"
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    print(f"✅ Installed packages ({len(installed_packages)}):")
    for package in installed_packages:
        print(f"   - {package}")
    
    if missing_packages:
        print(f"❌ Missing packages ({len(missing_packages)}):")
        for package in missing_packages:
            print(f"   - {package}")
    
    return len(missing_packages) == 0

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("\n🧪 Testing Model Loading...")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("Loading model...")
        model = AutoModelForTokenClassification.from_pretrained("models/results/checkpoint-2142")
        print("✅ Model loaded successfully")
        
        # Check model info
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test inference
        print("Testing inference...")
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
        print(f"❌ Error testing model: {str(e)}")
        return False

def main():
    """Main function to check model status"""
    print("🔍 Legal Document Analyzer - Model Status Check")
    print("="*60)
    
    # Run all checks
    checks = [
        ("Model Files", check_model_files),
        ("Model Configuration", check_model_config),
        ("Training Logs", check_training_logs),
        ("Device Compatibility", check_device_compatibility),
        ("Dependencies", check_dependencies),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {str(e)}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your model is ready to use.")
        print("\nNext steps:")
        print("1. Run: python test_trained_model.py")
        print("2. Check results in: results/trained_model_test_results.json")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")

if __name__ == "__main__":
    main() 