#!/usr/bin/env python3
"""
Fix Model Configuration - Update config to match actual training labels
"""

import os
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_model_config(checkpoint_path="models/results/checkpoint-2142"):
    """Fix the model configuration to match the actual training"""
    print("🔧 Fixing Model Configuration...")
    print("="*50)
    
    try:
        # Load the current config
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"Current config - num_labels: {config_data.get('num_labels', 'Not found')}")
        
        # Update the configuration for binary classification
        config_data['num_labels'] = 2  # Binary classification (0, 1)
        
        # Create label mappings
        config_data['id2label'] = {
            0: "O",      # Outside any clause
            1: "CLAUSE"  # Inside a clause
        }
        config_data['label2id'] = {
            "O": 0,
            "CLAUSE": 1
        }
        
        # Save the updated config
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("✅ Updated config.json with correct label configuration")
        print(f"   - num_labels: {config_data['num_labels']}")
        print(f"   - id2label: {config_data['id2label']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing config: {str(e)}")
        return False

def test_fixed_model():
    """Test if the fixed model can be loaded"""
    print("\n🧪 Testing Fixed Model...")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch
        
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
        print(f"Labels: {model.config.id2label}")
        
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
            probabilities = torch.softmax(outputs.logits, dim=2)
        
        print("✅ Inference test successful")
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Output shape: {outputs.logits.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # Show sample predictions
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        pred_labels = predictions[0].cpu().numpy()
        
        print("\nSample predictions:")
        for i, (token, pred) in enumerate(zip(tokens, pred_labels)):
            if token not in tokenizer.special_tokens_map.values():
                label = model.config.id2label.get(pred, "UNKNOWN")
                print(f"  {token}: {label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed model: {str(e)}")
        return False

def create_simple_test_script():
    """Create a simple test script for the fixed model"""
    script_content = '''#!/usr/bin/env python3
"""
Simple Test Script for Fixed Model
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pdfplumber
import re

def test_model_on_pdf(pdf_path, model_path="models/results/checkpoint-2142"):
    """Test the model on a PDF file"""
    print(f"Testing model on: {pdf_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\\n"
    
    # Clean text
    text = re.sub(r'\\s+', ' ', text).strip()
    print(f"Extracted {len(text)} characters")
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        probabilities = torch.softmax(outputs.logits, dim=2)
    
    # Extract clauses
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    pred_labels = predictions[0].cpu().numpy()
    confidences = probabilities[0].max(dim=1).values.cpu().numpy()
    
    clauses = []
    current_clause = {"text": "", "confidence": []}
    
    for token, pred, conf in zip(tokens, pred_labels, confidences):
        if token in tokenizer.special_tokens_map.values():
            continue
            
        label = model.config.id2label.get(pred, "UNKNOWN")
        
        if label == "CLAUSE":
            current_clause["text"] += token.replace("##", "")
            current_clause["confidence"].append(conf)
        else:
            if current_clause["text"] and len(current_clause["text"]) > 10:
                avg_conf = sum(current_clause["confidence"]) / len(current_clause["confidence"])
                clauses.append({
                    "text": current_clause["text"].strip(),
                    "confidence": avg_conf
                })
            current_clause = {"text": "", "confidence": []}
    
    # Add final clause if exists
    if current_clause["text"] and len(current_clause["text"]) > 10:
        avg_conf = sum(current_clause["confidence"]) / len(current_clause["confidence"])
        clauses.append({
            "text": current_clause["text"].strip(),
            "confidence": avg_conf
        })
    
    print(f"Found {len(clauses)} clauses:")
    for i, clause in enumerate(clauses[:5]):  # Show first 5
        print(f"  {i+1}. Confidence: {clause['confidence']:.3f}")
        print(f"     Text: {clause['text'][:100]}...")
    
    return clauses

if __name__ == "__main__":
    # Test on a sample PDF
    import os
    
    # Find a PDF file
    pdf_files = []
    for root, dirs, files in os.walk("full_contract_pdf"):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                break
        if pdf_files:
            break
    
    if pdf_files:
        test_model_on_pdf(pdf_files[0])
    else:
        print("No PDF files found in full_contract_pdf directory")
'''
    
    with open("simple_model_test.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created simple_model_test.py")

def main():
    """Main function to fix model configuration"""
    print("🔧 Legal Document Analyzer - Fix Model Configuration")
    print("="*60)
    
    # Fix the configuration
    if fix_model_config():
        print("\n✅ Configuration fixed successfully!")
        
        # Test the fixed model
        if test_fixed_model():
            print("\n🎉 Model is now ready to use!")
            
            # Create simple test script
            create_simple_test_script()
            
            print("\nNext steps:")
            print("1. Run: python check_model_status.py")
            print("2. Run: python test_trained_model.py")
            print("3. Run: python simple_model_test.py")
        else:
            print("\n❌ Model testing failed. Please check the error messages.")
    else:
        print("\n❌ Failed to fix configuration. Please check the error messages.")

if __name__ == "__main__":
    main() 