#!/usr/bin/env python3
"""
Inspect Model Weights - Check the actual model configuration from weights
"""

import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_model_weights(checkpoint_path="models/results/checkpoint-2142"):
    """Inspect the model weights to understand the actual configuration"""
    print("🔍 Inspecting Model Weights...")
    print("="*50)
    
    try:
        # Load the model weights directly
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            print(f"❌ Model weights not found: {model_path}")
            return False
        
        # Load the state dict from safetensors
        from safetensors import safe_open
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        print(f"✅ Loaded model weights with {len(state_dict)} layers")
        
        # Find the classification head
        classification_layers = {}
        for key, tensor in state_dict.items():
            if 'classifier' in key or 'classifier' in key.lower():
                classification_layers[key] = tensor.shape
                print(f"  {key}: {tensor.shape}")
        
        # Look for the final classification layer
        for key, shape in classification_layers.items():
            if 'bias' in key and len(shape) == 1:
                num_labels = shape[0]
                print(f"\n🎯 Detected {num_labels} labels in the model")
                
                # Update the config
                config_path = os.path.join(checkpoint_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    config_data['num_labels'] = num_labels
                    
                    # Create generic label mappings
                    config_data['id2label'] = {i: f"LABEL_{i}" for i in range(num_labels)}
                    config_data['label2id'] = {f"LABEL_{i}": i for i in range(num_labels)}
                    
                    # Save updated config
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    
                    print(f"✅ Updated config.json with {num_labels} labels")
                    print(f"   - id2label: {config_data['id2label']}")
                    
                    return True
        
        print("❌ Could not determine number of labels from weights")
        return False
        
    except Exception as e:
        print(f"❌ Error inspecting weights: {str(e)}")
        return False

def test_model_with_correct_config():
    """Test the model with the correct configuration"""
    print("\n🧪 Testing Model with Correct Configuration...")
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
        print(f"❌ Error testing model: {str(e)}")
        return False

def create_advanced_test_script():
    """Create an advanced test script that handles multiple label types"""
    script_content = '''#!/usr/bin/env python3
"""
Advanced Test Script for Multi-Label Model
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pdfplumber
import re
import json

def test_model_on_pdf(pdf_path, model_path="models/results/checkpoint-2142"):
    """Test the model on a PDF file with advanced clause extraction"""
    print(f"Testing model on: {pdf_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model has {model.config.num_labels} label types:")
    for label_id, label_name in model.config.id2label.items():
        print(f"  {label_id}: {label_name}")
    
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
    
    # Process text in chunks
    max_length = 512
    stride = 128
    
    all_clauses = []
    
    # Split text into overlapping chunks
    for i in range(0, len(text), max_length - stride):
        chunk_text = text[i:i + max_length]
        
        # Tokenize and predict
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.softmax(outputs.logits, dim=2)
        
        # Extract clauses from this chunk
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        pred_labels = predictions[0].cpu().numpy()
        confidences = probabilities[0].max(dim=1).values.cpu().numpy()
        
        # Group tokens by label
        current_clauses = {}
        for token, pred, conf in zip(tokens, pred_labels, confidences):
            if token in tokenizer.special_tokens_map.values():
                continue
                
            label = model.config.id2label.get(pred, "UNKNOWN")
            
            if label not in current_clauses:
                current_clauses[label] = {"text": "", "confidence": []}
            
            current_clauses[label]["text"] += token.replace("##", "")
            current_clauses[label]["confidence"].append(conf)
        
        # Add clauses to results
        for label, clause_data in current_clauses.items():
            if clause_data["text"] and len(clause_data["text"]) > 10:
                avg_conf = sum(clause_data["confidence"]) / len(clause_data["confidence"])
                all_clauses.append({
                    "label": label,
                    "text": clause_data["text"].strip(),
                    "confidence": avg_conf
                })
    
    # Group clauses by label
    clauses_by_label = {}
    for clause in all_clauses:
        label = clause["label"]
        if label not in clauses_by_label:
            clauses_by_label[label] = []
        clauses_by_label[label].append(clause)
    
    print(f"\\nFound clauses by type:")
    for label, clauses in clauses_by_label.items():
        print(f"  {label}: {len(clauses)} clauses")
        for i, clause in enumerate(clauses[:3]):  # Show first 3
            print(f"    {i+1}. Confidence: {clause['confidence']:.3f}")
            print(f"       Text: {clause['text'][:100]}...")
    
    return clauses_by_label

def save_results(results, output_file="results/advanced_model_test_results.json"):
    """Save test results to JSON file"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to serializable format
    serializable_results = {}
    for label, clauses in results.items():
        serializable_results[label] = [
            {
                "text": clause["text"][:200] + "..." if len(clause["text"]) > 200 else clause["text"],
                "confidence": float(clause["confidence"])
            }
            for clause in clauses
        ]
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

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
        results = test_model_on_pdf(pdf_files[0])
        save_results(results)
    else:
        print("No PDF files found in full_contract_pdf directory")
'''
    
    with open("advanced_model_test.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created advanced_model_test.py")

def main():
    """Main function to inspect and fix model weights"""
    print("🔍 Legal Document Analyzer - Inspect Model Weights")
    print("="*60)
    
    # Inspect the model weights
    if inspect_model_weights():
        print("\n✅ Model weights inspected successfully!")
        
        # Test the model with correct configuration
        if test_model_with_correct_config():
            print("\n🎉 Model is now ready to use!")
            
            # Create advanced test script
            create_advanced_test_script()
            
            print("\nNext steps:")
            print("1. Run: python check_model_status.py")
            print("2. Run: python advanced_model_test.py")
            print("3. Check results in: results/advanced_model_test_results.json")
        else:
            print("\n❌ Model testing failed. Please check the error messages.")
    else:
        print("\n❌ Failed to inspect model weights. Please check the error messages.")

if __name__ == "__main__":
    main() 