#!/usr/bin/env python3
"""
Test Trained Model - Comprehensive Testing Script
Tests the trained legal document analyzer model with real contract PDFs
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append('src')

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from src.data_loader import LegalDataLoaderFactory
from src.legal_tokenizer import LegalTokenizer
from src.evaluate_metrics import compute_metrics
import pdfplumber
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainedModelTester:
    def __init__(self, model_path="models/results/checkpoint-2142"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Number of labels: {len(self.id2label)}")
            logger.info(f"Labels: {list(self.id2label.values())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Clean text
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                return text
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            return None
    
    def predict_clauses(self, text, max_length=512, stride=128):
        """Predict clauses in the given text"""
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                return_overflowing_tokens=True,
                max_length=max_length,
                stride=stride
            )
            
            clauses = []
            
            # Process each chunk
            for chunk_idx in range(len(inputs["input_ids"])):
                chunk_inputs = {
                    k: v[chunk_idx:chunk_idx+1].to(self.device) 
                    for k, v in inputs.items()
                }
                
                # Model inference
                with torch.no_grad():
                    outputs = self.model(**chunk_inputs)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=2)
                    probabilities = torch.softmax(logits, dim=2)
                
                # Extract clauses from this chunk
                chunk_clauses = self.extract_clauses_from_chunk(
                    predictions[0], probabilities[0], 
                    inputs["input_ids"][chunk_idx]
                )
                clauses.extend(chunk_clauses)
            
            # Merge overlapping clauses
            return self.merge_overlapping_clauses(clauses)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return []
    
    def extract_clauses_from_chunk(self, predictions, probabilities, input_ids):
        """Extract clauses from a single chunk"""
        clauses = []
        current_clause = {"label": None, "text": "", "confidence": []}
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for token_idx, (token, pred, prob) in enumerate(zip(tokens, predictions, probabilities)):
            # Skip special tokens
            if token in self.tokenizer.special_tokens_map.values():
                continue
                
            label_id = pred.item()
            label = self.id2label.get(label_id, "UNKNOWN")
            confidence = prob[label_id].item()
            
            # Confidence thresholding
            if confidence < 0.5:
                label = "O"
            
            # Group contiguous tokens with same label
            if label != "O" and label == current_clause["label"]:
                current_clause["text"] += token.replace("##", "")
                current_clause["confidence"].append(confidence)
            else:
                # Save previous clause if it meets criteria
                if (current_clause["label"] and 
                    current_clause["label"] != "O" and 
                    len(current_clause["text"]) > 20 and
                    sum(current_clause["confidence"]) / len(current_clause["confidence"]) > 0.6):
                    
                    clauses.append({
                        "clause_type": current_clause["label"],
                        "text": current_clause["text"].strip(),
                        "confidence": sum(current_clause["confidence"]) / len(current_clause["confidence"])
                    })
                
                # Start new clause
                if label != "O":
                    current_clause = {
                        "label": label, 
                        "text": token.replace("##", ""), 
                        "confidence": [confidence]
                    }
                else:
                    current_clause = {"label": None, "text": "", "confidence": []}
        
        return clauses
    
    def merge_overlapping_clauses(self, clauses):
        """Merge overlapping clauses from different chunks"""
        if not clauses:
            return clauses
        
        # Sort by clause type and text
        clauses.sort(key=lambda x: (x["clause_type"], x["text"]))
        
        merged = []
        current = clauses[0]
        
        for clause in clauses[1:]:
            if (clause["clause_type"] == current["clause_type"] and
                clause["text"] in current["text"] or current["text"] in clause["text"]):
                # Merge overlapping clauses
                if len(clause["text"]) > len(current["text"]):
                    current = clause
            else:
                merged.append(current)
                current = clause
        
        merged.append(current)
        return merged
    
    def test_single_pdf(self, pdf_path):
        """Test the model on a single PDF file"""
        logger.info(f"Testing model on: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        logger.info(f"Extracted {len(text)} characters of text")
        
        # Predict clauses
        clauses = self.predict_clauses(text)
        
        # Create results
        results = {
            "pdf_path": pdf_path,
            "text_length": len(text),
            "clauses_found": len(clauses),
            "clauses": clauses,
            "summary": self.generate_summary(clauses)
        }
        
        return results
    
    def generate_summary(self, clauses):
        """Generate a summary of found clauses"""
        if not clauses:
            return "No clauses detected"
        
        clause_counts = {}
        for clause in clauses:
            clause_type = clause["clause_type"]
            clause_counts[clause_type] = clause_counts.get(clause_type, 0) + 1
        
        summary = f"Found {len(clauses)} clauses:\n"
        for clause_type, count in clause_counts.items():
            summary += f"- {clause_type}: {count}\n"
        
        return summary
    
    def test_multiple_pdfs(self, pdf_folder="full_contract_pdf"):
        """Test the model on multiple PDF files"""
        results = []
        
        for root, dirs, files in os.walk(pdf_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    
                    try:
                        result = self.test_single_pdf(pdf_path)
                        if result:
                            results.append(result)
                            logger.info(f"✓ Processed: {file}")
                        else:
                            logger.warning(f"✗ Failed: {file}")
                    except Exception as e:
                        logger.error(f"✗ Error processing {file}: {str(e)}")
        
        return results
    
    def save_results(self, results, output_file="results/trained_model_test_results.json"):
        """Save test results to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                "pdf_path": result["pdf_path"],
                "text_length": result["text_length"],
                "clauses_found": result["clauses_found"],
                "summary": result["summary"],
                "clauses": [
                    {
                        "clause_type": clause["clause_type"],
                        "text": clause["text"][:200] + "..." if len(clause["text"]) > 200 else clause["text"],
                        "confidence": float(clause["confidence"])
                    }
                    for clause in result["clauses"]
                ]
            }
            serializable_results.append(serializable_result)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_detailed_results(self, results):
        """Print detailed results"""
        print("\n" + "="*80)
        print("TRAINED MODEL TEST RESULTS")
        print("="*80)
        
        total_pdfs = len(results)
        total_clauses = sum(r["clauses_found"] for r in results)
        
        print(f"Total PDFs processed: {total_pdfs}")
        print(f"Total clauses found: {total_clauses}")
        print(f"Average clauses per PDF: {total_clauses/total_pdfs:.1f}")
        
        # Clause type distribution
        clause_types = {}
        for result in results:
            for clause in result["clauses"]:
                clause_type = clause["clause_type"]
                clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
        
        print(f"\nClause type distribution:")
        for clause_type, count in sorted(clause_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {clause_type}: {count}")
        
        # Show sample results
        print(f"\nSample results:")
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"\n{i+1}. {os.path.basename(result['pdf_path'])}")
            print(f"   Clauses found: {result['clauses_found']}")
            print(f"   Summary: {result['summary']}")
            
            # Show first few clauses
            for j, clause in enumerate(result['clauses'][:2]):
                print(f"   - {clause['clause_type']} (confidence: {clause['confidence']:.2f})")
                print(f"     Text: {clause['text'][:100]}...")

def main():
    """Main function to test the trained model"""
    print("🔍 Testing Trained Legal Document Analyzer Model")
    print("="*60)
    
    # Initialize tester
    tester = TrainedModelTester()
    
    # Load model
    if not tester.load_model():
        print("❌ Failed to load model. Please check the model path.")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test on multiple PDFs
    print("\n📄 Testing on contract PDFs...")
    results = tester.test_multiple_pdfs()
    
    if not results:
        print("❌ No PDFs found or processed successfully.")
        return
    
    # Save results
    tester.save_results(results)
    
    # Print detailed results
    tester.print_detailed_results(results)
    
    print("\n🎉 Testing completed successfully!")
    print("Check 'results/trained_model_test_results.json' for detailed results.")

if __name__ == "__main__":
    main() 