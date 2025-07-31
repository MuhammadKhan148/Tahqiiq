import os, json, pandas as pd, torch
import ast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.utils import load_cuad_dataset, extract_text_from_pdf, predict_clauses
from tabulate import tabulate

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 32

class LegalBertTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Tokenizer loaded: {MODEL_NAME}")

    def parse_labels(self, label_str):
        """Parse string representation of list into actual list"""
        try:
            # Handle string representation of lists like "[]" or "[1,2,3]"
            if isinstance(label_str, str):
                return ast.literal_eval(label_str)
            return label_str
        except (ValueError, SyntaxError):
            # If parsing fails, return empty list as fallback
            return []

    def encode_batch(self, texts, labels=None):
        enc = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        out = {"input_ids": enc["input_ids"],
               "attention_mask": enc["attention_mask"]}
        if labels is not None:
            # Convert labels to appropriate format
            processed_labels = []
            for label in labels:
                parsed_label = self.parse_labels(label)
                # For multi-label classification, you might want to convert to binary vector
                # For now, we'll store the length of the label list as a simple numeric target
                processed_labels.append(len(parsed_label))
            out["labels"] = torch.tensor(processed_labels, dtype=torch.long)
        return out

    def tokenize_file(self, csv_in, out_dir):
        if not os.path.exists(csv_in):
            raise FileNotFoundError(f"{csv_in} not found.")
        os.makedirs(out_dir, exist_ok=True)

        df = pd.read_csv(csv_in)
        if {"text", "labels"} - set(df.columns):
            raise ValueError("CSV must contain 'text' and 'labels' columns.")

        # Parse all labels upfront to get statistics
        df['parsed_labels'] = df['labels'].apply(self.parse_labels)
        
        ids, masks, labels = [], [], []
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Tokenizing"):
            batch = df.iloc[i:i+BATCH_SIZE]
            enc = self.encode_batch(batch["text"].tolist(), batch["labels"].tolist())
            ids.append(enc["input_ids"])
            masks.append(enc["attention_mask"])
            labels.append(enc["labels"])

        torch.save(torch.cat(ids),   os.path.join(out_dir, "input_ids.pt"))
        torch.save(torch.cat(masks), os.path.join(out_dir, "attention_masks.pt"))
        torch.save(torch.cat(labels),os.path.join(out_dir, "labels.pt"))

        # Create metadata with more detailed label information
        label_lengths = df['parsed_labels'].apply(len)
        unique_labels = set()
        for label_list in df['parsed_labels']:
            unique_labels.update(label_list)
        
        meta = {
            "num_samples": int(len(df)),
            "max_length": MAX_LEN,
            "model_name": MODEL_NAME,
            "unique_labels": sorted(list(unique_labels)),
            "label_length_distribution": {str(k): int(v) for k, v in label_lengths.value_counts().to_dict().items()},
            "avg_labels_per_sample": float(label_lengths.mean()),
            "max_labels_per_sample": int(label_lengths.max()),
            "empty_label_count": int((label_lengths == 0).sum())
        }
        
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"Saved tokenized tensors & metadata to {out_dir}")
        print("Label statistics:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Empty labels: {meta['empty_label_count']}")
        print(f"   - Unique labels found: {len(unique_labels)}")
        print(f"   - Average labels per sample: {meta['avg_labels_per_sample']:.2f}")
        return meta
    
    def predict_clauses_from_pdf(self, pdf_path: str, model_path: str = "./models/fine_tuned_legalbert_cuad"):
        """Predict legal clauses from PDF using trained model"""
        print(f"Predicting clauses from PDF: {pdf_path}")
        
        # Load labels
        _, _, _, _, id2label = load_cuad_dataset()
        
        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Extract text from PDF
        contract_text = extract_text_from_pdf(pdf_path)
        
        # Predict clauses
        detected_clauses = predict_clauses(contract_text, model, tokenizer, id2label)
        
        # Display in table
        if detected_clauses:
            table_data = [[clause["clause_type"], f"{clause['confidence']:.4f}", clause["text"][:200] + "..."] for clause in detected_clauses]
            headers = ["Clause Type", "Confidence", "Text Snippet"]
            print(tabulate(table_data, headers, tablefmt="grid"))
        else:
            print("No clauses detected in the document.")
        
        return detected_clauses

def main():
    input_csv = "./data/processed/unfair_tos_test_cleaned.csv"
    out_dir   = "./data/tokenized/"
    tokenizer = LegalBertTokenizer()
    tokenizer.tokenize_file(input_csv, out_dir)
    
    # Example of PDF clause prediction
    # tokenizer.predict_clauses_from_pdf("./data/contract.pdf")

if __name__ == "__main__":
    main()