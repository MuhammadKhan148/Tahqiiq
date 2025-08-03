#!/usr/bin/env python3
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
                text += page_text + "\n"
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
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
    
    print(f"\nFound clauses by type:")
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
