#!/usr/bin/env python3
"""
Training-only script for legal document analyzer
Use this on your training PC after transferring the project
"""

import sys
sys.path.append('.')

def train_model():
    """Train the CUAD model"""
    print("="*60)
    print("LEGAL DOCUMENT ANALYZER - MODEL TRAINING")
    print("="*60)
    
    try:
        from src.data_loader import LegalDataLoaderFactory
        
        print("Loading prepared CUAD datasets...")
        factory = LegalDataLoaderFactory()
        tokenized_datasets = factory.load_cuad_datasets()
        
        print(f"✓ Loaded {len(tokenized_datasets['train'])} training samples")
        print(f"✓ Loaded {len(tokenized_datasets['validation'])} validation samples")
        
        print("\nStarting model training...")
        print("⚠️  This may take several hours depending on your hardware")
        print("⚠️  The model will be saved to: ./models/fine_tuned_legalbert_cuad/")
        
        trainer = factory.train_cuad_model(tokenized_datasets)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your trained model is ready!")
        print("Model saved to: ./models/fine_tuned_legalbert_cuad/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
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
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies are installed!")
    return True

def main():
    """Main training function"""
    print("Legal Document Analyzer - Training Script")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return
    
    # Check if data is ready
    import os
    if not os.path.exists("data/tokenized_cuad/train"):
        print("❌ Training data not found!")
        print("Please run the setup first: python run_workflow.py")
        return
    
    # Start training
    success = train_model()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("You can now:")
        print("1. Test the model: python test_pdf_prediction.py")
        print("2. Upload to GitHub: git add . && git commit -m 'Trained model' && git push")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main() 