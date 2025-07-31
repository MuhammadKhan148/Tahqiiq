#!/usr/bin/env python3
"""
Test current status of the legal document analyzer project
"""

import os
from pathlib import Path

def check_files():
    """Check what files have been created"""
    print("Checking current project status...")
    print("="*50)
    
    # Check CUAD dataset
    cuad_local = Path("data/cuad_local/CUAD_v1.json")
    if cuad_local.exists():
        print(f"✓ CUAD dataset: {cuad_local}")
    else:
        print(f"❌ CUAD dataset missing: {cuad_local}")
    
    # Check tokenized data
    tokenized_dir = Path("data/tokenized_cuad")
    if tokenized_dir.exists():
        train_dir = tokenized_dir / "train"
        val_dir = tokenized_dir / "validation"
        if train_dir.exists() and val_dir.exists():
            print(f"✓ Tokenized CUAD data: {tokenized_dir}")
        else:
            print(f"❌ Tokenized data incomplete: {tokenized_dir}")
    else:
        print(f"❌ Tokenized data missing: {tokenized_dir}")
    
    # Check trained model
    model_dir = Path("models/fine_tuned_legalbert_cuad")
    if model_dir.exists():
        print(f"✓ Trained model: {model_dir}")
    else:
        print(f"❌ Trained model missing: {model_dir}")
    
    # Check test PDF
    test_pdf = Path("data/contract.pdf")
    if test_pdf.exists():
        print(f"✓ Test PDF: {test_pdf}")
    else:
        print(f"❌ Test PDF missing: {test_pdf}")
    
    print("\n" + "="*50)
    print("Summary:")
    
    if cuad_local.exists() and tokenized_dir.exists():
        print("✓ Dataset preparation: COMPLETE")
    else:
        print("❌ Dataset preparation: INCOMPLETE")
    
    if model_dir.exists():
        print("✓ Model training: COMPLETE")
    else:
        print("❌ Model training: INCOMPLETE")
    
    if test_pdf.exists():
        print("✓ Test setup: COMPLETE")
    else:
        print("❌ Test setup: INCOMPLETE")

def test_imports():
    """Test if all imports work"""
    print("\nTesting imports...")
    print("="*30)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets: {e}")
    
    try:
        import evaluate
        print(f"✓ Evaluate: {evaluate.__version__}")
    except ImportError as e:
        print(f"❌ Evaluate: {e}")
    
    try:
        import pdfplumber
        print(f"✓ PDFPlumber: {pdfplumber.__version__}")
    except ImportError as e:
        print(f"❌ PDFPlumber: {e}")

def test_cuad_loading():
    """Test CUAD dataset loading"""
    print("\nTesting CUAD dataset loading...")
    print("="*40)
    
    try:
        import sys
        sys.path.append('.')
        from src.utils import load_cuad_dataset
        
        dataset, label_list, num_labels, label2id, id2label = load_cuad_dataset()
        print(f"✓ CUAD dataset loaded successfully")
        print(f"  - Number of labels: {num_labels}")
        print(f"  - Sample labels: {label_list[:5]}")
        
    except Exception as e:
        print(f"❌ CUAD dataset loading failed: {e}")

if __name__ == "__main__":
    check_files()
    test_imports()
    test_cuad_loading() 