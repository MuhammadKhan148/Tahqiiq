#!/usr/bin/env python3
"""
Test PDF prediction functionality
"""

import sys
sys.path.append('.')

def test_pdf_extraction():
    """Test PDF text extraction"""
    print("Testing PDF text extraction...")
    print("="*40)
    
    try:
        from src.utils import extract_text_from_pdf
        
        pdf_path = "data/contract.pdf"
        text = extract_text_from_pdf(pdf_path)
        
        print(f"✓ PDF text extraction successful")
        print(f"  - PDF: {pdf_path}")
        print(f"  - Text length: {len(text)} characters")
        print(f"  - First 200 chars: {text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF text extraction failed: {e}")
        return False

def test_cuad_utils():
    """Test CUAD utilities"""
    print("\nTesting CUAD utilities...")
    print("="*30)
    
    try:
        from src.utils import load_cuad_dataset
        
        dataset, label_list, num_labels, label2id, id2label = load_cuad_dataset()
        
        print(f"✓ CUAD utilities working")
        print(f"  - Number of labels: {num_labels}")
        print(f"  - Sample labels: {label_list[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUAD utilities failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\nTesting tokenizer...")
    print("="*25)
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        sample_text = "This is a test contract with parties and agreement date."
        tokens = tokenizer(sample_text, return_tensors="pt")
        
        print(f"✓ Tokenizer working")
        print(f"  - Sample text: {sample_text}")
        print(f"  - Token IDs shape: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Legal Document Analyzer Components")
    print("="*50)
    
    # Test PDF extraction
    pdf_ok = test_pdf_extraction()
    
    # Test CUAD utilities
    cuad_ok = test_cuad_utils()
    
    # Test tokenizer
    tokenizer_ok = test_tokenizer()
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  - PDF extraction: {'✓' if pdf_ok else '❌'}")
    print(f"  - CUAD utilities: {'✓' if cuad_ok else '❌'}")
    print(f"  - Tokenizer: {'✓' if tokenizer_ok else '❌'}")
    
    if pdf_ok and cuad_ok and tokenizer_ok:
        print("\n🎉 All core components are working!")
        print("The system is ready for model training and inference.")
    else:
        print("\n⚠️  Some components need attention.")

if __name__ == "__main__":
    main() 