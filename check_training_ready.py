#!/usr/bin/env python3
"""
Check if the project is ready for training on another PC
"""

import os
from pathlib import Path

def check_training_readiness():
    """Check if everything is ready for training"""
    print("Legal Document Analyzer - Training Readiness Check")
    print("="*60)
    
    # Check essential files
    essential_files = [
        "data/cuad_local/CUAD_v1.json",
        "data/tokenized_cuad/train",
        "data/tokenized_cuad/validation",
        "data/contract.pdf",
        "src/utils.py",
        "src/data_loader.py",
        "requirements.txt"
    ]
    
    print("Checking essential files...")
    all_files_ok = True
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_ok = False
    
    # Check data sizes
    print("\nChecking data sizes...")
    
    if os.path.exists("data/tokenized_cuad/train"):
        train_size = len(os.listdir("data/tokenized_cuad/train"))
        print(f"✓ Training data: {train_size} files")
    
    if os.path.exists("data/tokenized_cuad/validation"):
        val_size = len(os.listdir("data/tokenized_cuad/validation"))
        print(f"✓ Validation data: {val_size} files")
    
    # Check CUAD dataset size
    if os.path.exists("data/cuad_local/CUAD_v1.json"):
        size_mb = os.path.getsize("data/cuad_local/CUAD_v1.json") / (1024 * 1024)
        print(f"✓ CUAD dataset: {size_mb:.1f} MB")
    
    # Check if model already exists
    if os.path.exists("models/fine_tuned_legalbert_cuad"):
        print("⚠️  Trained model already exists!")
        print("   If you want to retrain, delete: models/fine_tuned_legalbert_cuad/")
    
    print("\n" + "="*60)
    
    if all_files_ok:
        print("🎉 PROJECT IS READY FOR TRAINING!")
        print("="*60)
        print("You can now:")
        print("1. Transfer this project to your training PC")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run training: python train_only.py")
        print("4. After training, test: python test_pdf_prediction.py")
    else:
        print("❌ PROJECT IS NOT READY FOR TRAINING")
        print("="*60)
        print("Please run: python run_workflow.py")
        print("This will prepare all missing files.")

def check_dependencies():
    """Check if dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'evaluate', 
        'seqeval', 'pdfplumber', 'tabulate', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies are installed!")
    return True

def main():
    """Main function"""
    check_training_readiness()
    check_dependencies()

if __name__ == "__main__":
    main() 