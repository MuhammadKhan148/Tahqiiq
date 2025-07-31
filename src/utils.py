import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset
from sklearn.metrics import classification_report
import pdfplumber
import os
import json
from typing import Dict, List, Optional, Tuple

# Load dataset and get labels (CUAD integration)
def load_cuad_dataset():
    """Load CUAD dataset and return labels for legal clause classification"""
    try:
        # Try to load from local CUAD file
        dataset = load_dataset("json", data_files="./data/cuad_local/CUAD_v1.json", field="data")
    except:
        # Fallback to HuggingFace hub
        dataset = load_dataset("cuad")
    
    clause_types = [
        "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
        "Renewal Term", "Notice to Terminate Renewal", "Governing Law", "Most Favored Nation",
        "Non-Compete", "Exclusivity", "No-Solicit of Customers", "Competitive Restriction Exception",
        "No-Solicit of Employees", "Non-Disparagement", "Termination for Convenience",
        "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)", "Change of Control",
        "Anti-Assignment", "Revenue/Profit Sharing", "Price Restriction", "Minimum Commitment",
        "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership", "License Grant",
        "Non-Transferable License", "Affiliate IP License-Licensor", "Affiliate IP License-Licensee",
        "Unlimited/All-You-Can-Eat License", "Irrevocable or Perpetual License", "Source Code Escrow",
        "Post-Termination Services", "Audit Rights", "Uncapped Liability", "Cap on Liability",
        "Liquidated Damages", "Warranty Duration", "Insurance", "Covenant not to Sue",
        "Third Party Beneficiary"
    ]
    label_list = ["O"] + clause_types
    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    return dataset, label_list, num_labels, label2id, id2label

# Compute overall metrics using seqeval
def compute_metrics(p, id2label):
    """Compute token classification metrics using seqeval"""
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [id2label[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Detailed per-clause metrics using sklearn
def detailed_compute_metrics(p, label_list, id2label):
    """Compute detailed per-clause classification metrics"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_preds = [pred for prediction, label in zip(predictions, labels) for (pred, lbl) in zip(prediction, label) if lbl != -100]
    true_lbls = [lbl for prediction, label in zip(predictions, labels) for (pred, lbl) in zip(prediction, label) if lbl != -100]
    
    report = classification_report(true_lbls, true_preds, target_names=label_list, output_dict=True, zero_division=0)
    return report

# Data collator for token classification
def get_data_collator(tokenizer):
    """Get data collator for token classification tasks"""
    return DataCollatorForTokenClassification(tokenizer=tokenizer)

# Extract and clean text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract and clean text from PDF file"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # Simple cleaning
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip() and len(line) > 20)
    return text

# Predict clauses on text (for inference, groups contiguous non-O)
def predict_clauses(text, model, tokenizer, id2label, chunk_size=512, stride=128, device=None):
    """Predict legal clauses in text using token classification model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=False, return_overflowing_tokens=True, max_length=chunk_size, stride=stride)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    probabilities = torch.softmax(logits, dim=2)
    
    clauses = []
    current_clause = {"label": None, "text": "", "confidence": []}
    
    for chunk_idx in range(predictions.shape[0]):
        chunk_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][chunk_idx])
        chunk_preds = predictions[chunk_idx]
        chunk_probs = probabilities[chunk_idx]
        
        for token_idx, (token, pred, prob) in enumerate(zip(chunk_tokens, chunk_preds, chunk_probs)):
            if token in tokenizer.special_tokens_map.values():
                continue
            label_id = pred.item()
            label = id2label.get(label_id, "UNKNOWN")
            conf = prob[label_id].item()
            
            if label != "O" and label == current_clause["label"] and conf > 0.5:
                current_clause["text"] += token.replace("##", "")
                current_clause["confidence"].append(conf)
            else:
                if current_clause["label"] and current_clause["label"] != "O" and len(current_clause["text"]) > 20:
                    avg_conf = np.mean(current_clause["confidence"])
                    clauses.append({
                        "clause_type": current_clause["label"],
                        "text": current_clause["text"].strip(),
                        "confidence": avg_conf
                    })
                if label != "O":
                    current_clause = {"label": label, "text": token.replace("##", ""), "confidence": [conf]}
                else:
                    current_clause = {"label": None, "text": "", "confidence": []}
    
    # Add last clause
    if current_clause["label"] and current_clause["label"] != "O" and len(current_clause["text"]) > 20:
        avg_conf = np.mean(current_clause["confidence"])
        clauses.append({
            "clause_type": current_clause["label"],
            "text": current_clause["text"].strip(),
            "confidence": avg_conf
        })
    
    return clauses

# Enhanced data preparation for CUAD dataset
def prepare_cuad_dataset(tokenizer, dataset, label2id, max_length=512, stride=128):
    """Prepare CUAD dataset for token classification training"""
    processed_data = {"train": [], "validation": []}
    
    # Manual split if validation doesn't exist
    if "validation" not in dataset:
        full_dataset = dataset["train"]
        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
        dataset = {
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        }
    
    for split, ds in dataset.items():
        for example_idx, example in enumerate(ds):
            context = example["paragraphs"][0]["context"]
            qas = example["paragraphs"][0]["qas"]
            
            # Collect spans per clause type
            spans_per_type = {}
            for qa in qas:
                clause_type = qa["question"].split(":")[0].strip()
                if clause_type not in label2id:
                    continue
                spans = []
                for ans in qa["answers"]:
                    if ans["text"]:
                        start = ans["answer_start"]
                        end = start + len(ans["text"])
                        spans.append((start, end))
                spans_per_type[clause_type] = spans
            
            # Tokenize with offsets and overflowing chunks
            inputs = tokenizer(
                context,
                return_offsets_mapping=True,
                max_length=max_length,
                truncation=True,
                stride=stride,
                return_overflowing_tokens=True,
            )
            
            for chunk_idx in range(len(inputs["input_ids"])):
                chunk_input_ids = inputs["input_ids"][chunk_idx]
                chunk_attention_mask = inputs["attention_mask"][chunk_idx]
                chunk_offsets = inputs["offset_mapping"][chunk_idx]
                
                labels = [0] * len(chunk_input_ids)  # Default O
                
                for tok_idx, (off_start, off_end) in enumerate(chunk_offsets):
                    if off_start == off_end == 0:  # Special token
                        labels[tok_idx] = -100
                        continue
                    
                    assigned_label = 0  # O
                    for type_name, type_spans in spans_per_type.items():
                        label_id = label2id[type_name]
                        for span_start, span_end in type_spans:
                            if not (off_end <= span_start or off_start >= span_end):
                                assigned_label = label_id
                                break
                        if assigned_label != 0:
                            break
                    
                    labels[tok_idx] = assigned_label
                
                # Append chunk as a sample
                processed_data[split].append({
                    "input_ids": chunk_input_ids,
                    "attention_mask": chunk_attention_mask,
                    "labels": labels
                })
    
    return processed_data 