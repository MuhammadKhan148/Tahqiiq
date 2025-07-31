#!/usr/bin/env python3
"""
Complete workflow script to create all missing files for the legal document analyzer.
This script will:
1. Prepare CUAD dataset
2. Train the model
3. Evaluate the model
4. Test PDF clause prediction
"""

import os
import sys
import shutil
from pathlib import Path

def setup_files():
    """Setup required files and directories"""
    print("Setting up required files...")
    
    # Create cuad_local directory and copy CUAD dataset
    cuad_local_dir = Path("data/cuad_local")
    cuad_local_dir.mkdir(exist_ok=True)
    
    # Copy CUAD dataset to cuad_local
    source_cuad = Path("data/CUAD_v1.json")
    target_cuad = cuad_local_dir / "CUAD_v1.json"
    
    if source_cuad.exists() and not target_cuad.exists():
        shutil.copy2(source_cuad, target_cuad)
        print(f"✓ Copied CUAD dataset to {target_cuad}")
    elif target_cuad.exists():
        print(f"✓ CUAD dataset already exists at {target_cuad}")
    else:
        print(f"❌ CUAD dataset not found at {source_cuad}")
        return False
    
    # Copy a PDF for testing
    pdf_source = Path("data/raw/20181091110018405RightofAccesstoInformationAct2017.pdf")
    pdf_target = Path("data/contract.pdf")
    
    if pdf_source.exists() and not pdf_target.exists():
        shutil.copy2(pdf_source, pdf_target)
        print(f"✓ Copied PDF for testing to {pdf_target}")
    elif pdf_target.exists():
        print(f"✓ Test PDF already exists at {pdf_target}")
    else:
        print(f"❌ PDF source not found at {pdf_source}")
        return False
    
    return True

def prepare_cuad_dataset():
    """Prepare CUAD dataset for training"""
    print("\n" + "="*50)
    print("STEP 1: Preparing CUAD Dataset")
    print("="*50)
    
    try:
        import sys
        sys.path.append('.')
        from src.data_loader import LegalDataLoaderFactory
        
        factory = LegalDataLoaderFactory()
        tokenized_datasets = factory.prepare_cuad_dataset()
        
        print("✓ CUAD dataset prepared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error preparing CUAD dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model():
    """Train the CUAD model"""
    print("\n" + "="*50)
    print("STEP 2: Training CUAD Model")
    print("="*50)
    
    try:
        import sys
        sys.path.append('.')
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

def evaluate_model():
    """Evaluate the trained model"""
    print("\n" + "="*50)
    print("STEP 3: Evaluating Model")
    print("="*50)
    
    try:
        import sys
        sys.path.append('.')
        from src.evaluate_metrics import LegalBertEvaluator
        
        evaluator = LegalBertEvaluator()
        results = evaluator.evaluate_cuad_model()
        
        print("✓ Model evaluation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_prediction():
    """Test PDF clause prediction"""
    print("\n" + "="*50)
    print("STEP 4: Testing PDF Clause Prediction")
    print("="*50)
    
    try:
        import sys
        sys.path.append('.')
        from src.pdf_clause_predictor import PDFClausePredictor
        
        predictor = PDFClausePredictor()
        clauses = predictor.predict_from_pdf("data/contract.pdf")
        
        print("✓ PDF clause prediction completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in PDF prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete workflow"""
    print("Legal Document Analyzer - Complete Workflow")
    print("="*50)
    
    # Step 0: Setup files
    if not setup_files():
        print("❌ Setup failed. Exiting.")
        return
    
    # Step 1: Prepare dataset
    if not prepare_cuad_dataset():
        print("❌ Dataset preparation failed. Exiting.")
        return
    
    # Step 2: Train model
    if not train_model():
        print("❌ Model training failed. Exiting.")
        return
    
    # Step 3: Evaluate model
    if not evaluate_model():
        print("❌ Model evaluation failed. Exiting.")
        return
    
    # Step 4: Test PDF prediction
    if not test_pdf_prediction():
        print("❌ PDF prediction failed. Exiting.")
        return
    
    print("\n" + "="*50)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Your legal document analyzer is now ready to use!")
    print("\nCreated files:")
    print("- data/tokenized_cuad/ (tokenized dataset)")
    print("- models/fine_tuned_legalbert_cuad/ (trained model)")
    print("- results/ (evaluation results)")
    print("- data/contract.pdf (test PDF)")

if __name__ == "__main__":
    main() 