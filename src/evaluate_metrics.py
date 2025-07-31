import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)
from typing import Dict, List, Optional, Union
import torch
import evaluate
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from src.utils import compute_metrics, detailed_compute_metrics, load_cuad_dataset, get_data_collator

class LegalBertEvaluator:
    """Evaluation metrics calculator for LEGAL-BERT model predictions"""
    
    def __init__(self, splits_dir: str = "data/splits", results_dir: str = "results"):
        """Initialize evaluator with directories"""
        self.splits_dir = splits_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("Initializing LegalBertEvaluator...")
        print(f"Splits directory: {splits_dir}")
        print(f"Results directory: {results_dir}")
    
    def load_ground_truth(self, split_name: str = "test") -> Dict:
        """Load ground truth labels from test split"""
        print(f"Loading ground truth from {split_name} split...")
        
        # Load manifest to get true labels
        manifest_path = os.path.join(self.splits_dir, f"{split_name}_manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        ground_truth = []
        sample_ids = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                ground_truth.append(entry['label'])
                sample_ids.append(entry['id'])
        
        print(f"Loaded {len(ground_truth)} ground truth labels")
        
        return {
            'labels': np.array(ground_truth),
            'sample_ids': sample_ids,
            'split_name': split_name
        }
    
    def load_predictions(self, predictions_file: str) -> Dict:
        """Load model predictions from file"""
        print(f"Loading predictions from {predictions_file}...")
        
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        # Support multiple formats
        if predictions_file.endswith('.npy'):
            predictions = np.load(predictions_file)
            sample_ids = [f"sample_{i}" for i in range(len(predictions))]
        
        elif predictions_file.endswith('.csv'):
            df = pd.read_csv(predictions_file)
            if 'predicted_label' in df.columns:
                predictions = df['predicted_label'].values
            elif 'prediction' in df.columns:
                predictions = df['prediction'].values
            else:
                predictions = df.iloc[:, -1].values  # Last column
            
            sample_ids = df.get('sample_id', [f"sample_{i}" for i in range(len(predictions))]).tolist()
        
        elif predictions_file.endswith('.json'):
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                predictions = np.array(data)
                sample_ids = [f"sample_{i}" for i in range(len(predictions))]
            elif isinstance(data, dict):
                predictions = np.array(data['predictions'])
                sample_ids = data.get('sample_ids', [f"sample_{i}" for i in range(len(predictions))])
        
        else:
            raise ValueError(f"Unsupported file format: {predictions_file}")
        
        print(f"Loaded {len(predictions)} predictions")
        
        return {
            'predictions': predictions,
            'sample_ids': sample_ids
        }
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       class_names: Optional[List[str]] = None) -> Dict:
        """Compute comprehensive classification metrics"""
        
    def evaluate_cuad_model(self, model_path: str = "./models/fine_tuned_legalbert_cuad", 
                           data_path: str = "./data/tokenized_cuad") -> Dict:
        """Evaluate CUAD token classification model"""
        print("Evaluating CUAD token classification model...")
        
        # Load labels
        _, label_list, _, _, id2label = load_cuad_dataset()
        
        # Load tokenized dataset
        tokenized_datasets = {
            "validation": load_from_disk(os.path.join(data_path, "validation"))
        }
        
        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Dummy training args for Trainer
        training_args = TrainingArguments(
            output_dir="./models/results",
            per_device_eval_batch_size=16,
        )
        
        data_collator = get_data_collator(tokenizer)
        
        # Trainer for evaluation
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, id2label),
        )
        
        # Overall evaluation
        eval_results = trainer.evaluate()
        print("Overall Evaluation Results:")
        print(eval_results)
        
        # Detailed per-clause
        predictions = trainer.predict(tokenized_datasets["validation"])
        detailed_results = detailed_compute_metrics((predictions.predictions, predictions.label_ids), label_list, id2label)
        
        # Print sample per-clause F1 (skip O)
        print("Per-Clause F1-Scores (excluding O, sample first 5 clauses):")
        clause_metrics = {k: v for k, v in detailed_results.items() if k != "O" and isinstance(v, dict)}
        for label, metrics in list(clause_metrics.items())[:5]:
            print(f"{label}: F1 = {metrics['f1-score']:.4f}")
        
        return {
            "overall_results": eval_results,
            "detailed_results": detailed_results
        }
        print("Computing classification metrics...")
        
        # Handle edge case of single sample
        if len(y_true) == 1:
            print("Single sample evaluation - limited metrics available")
            accuracy = accuracy_score(y_true, y_pred)
            
            # For single sample, precision/recall are either 1 or 0
            correct = (y_true[0] == y_pred[0])
            
            return {
                'accuracy': float(accuracy),
                'single_sample_evaluation': True,
                'correct_prediction': bool(correct),
                'true_label': int(y_true[0]),
                'predicted_label': int(y_pred[0]),
                'total_samples': 1
            }
        
        # Basic metrics for multiple samples
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics with zero division handling
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Build results
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        if class_names is None:
            class_names = [f"Class_{label}" for label in unique_labels]
        elif len(class_names) < len(unique_labels):
            # Extend class names if needed
            class_names.extend([f"Class_{label}" for label in unique_labels[len(class_names):]])
        
        results = {
            'accuracy': float(accuracy),
            'macro_avg': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1_score': float(f1_macro)
            },
            'micro_avg': {
                'precision': float(precision_micro),
                'recall': float(recall_micro),
                'f1_score': float(f1_micro)
            },
            'per_class': {},
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'total_samples': len(y_true),
            'single_sample_evaluation': False
        }
        
        # Per-class metrics
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                results['per_class'][class_names[i]] = {
                    'precision': float(precision[i]) if not np.isnan(precision[i]) else 0.0,
                    'recall': float(recall[i]) if not np.isnan(recall[i]) else 0.0,
                    'f1_score': float(f1[i]) if not np.isnan(f1[i]) else 0.0,
                    'support': int(support[i])
                }
        
        print(f"Computed metrics - Accuracy: {accuracy:.3f}, Macro F1: {f1_macro:.3f}")
        
        return results
    
    def create_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: Optional[List[str]] = None) -> str:
        """Create detailed classification report"""
        if len(y_true) == 1:
            return f"Single sample evaluation:\nTrue: {y_true[0]}, Predicted: {y_pred[0]}, Correct: {y_true[0] == y_pred[0]}"
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in sorted(np.unique(np.concatenate([y_true, y_pred])))]
        
        return classification_report(y_true, y_pred, target_names=class_names, digits=3)
    
    def save_results(self, results: Dict, predictions_info: Dict, ground_truth_info: Dict, 
                    prefix: str = "evaluation") -> str:
        """Save evaluation results to files"""
        print("Saving evaluation results...")
        
        # Add metadata
        results['metadata'] = {
            'split_name': ground_truth_info['split_name'],
            'total_predictions': len(predictions_info['predictions']),
            'total_ground_truth': len(ground_truth_info['labels']),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Save as JSON
        json_path = os.path.join(self.results_dir, f"{prefix}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save simplified CSV for single sample
        if results.get('single_sample_evaluation', False):
            csv_data = [{
                'metric_type': 'single_sample',
                'accuracy': results['accuracy'],
                'correct_prediction': results['correct_prediction'],
                'true_label': results['true_label'],
                'predicted_label': results['predicted_label'],
                'total_samples': 1
            }]
        else:
            # Save as CSV (flattened metrics) for multiple samples
            csv_data = []
            
            # Overall metrics
            csv_data.append({
                'metric_type': 'overall',
                'class': 'all',
                'precision': results['micro_avg']['precision'],
                'recall': results['micro_avg']['recall'],
                'f1_score': results['micro_avg']['f1_score'],
                'accuracy': results['accuracy'],
                'support': results['total_samples']
            })
            
            # Macro averages
            csv_data.append({
                'metric_type': 'macro_avg',
                'class': 'all',
                'precision': results['macro_avg']['precision'],
                'recall': results['macro_avg']['recall'],
                'f1_score': results['macro_avg']['f1_score'],
                'accuracy': results['accuracy'],
                'support': results['total_samples']
            })
            
            # Per-class metrics
            for class_name, metrics in results.get('per_class', {}).items():
                csv_data.append({
                    'metric_type': 'per_class',
                    'class': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'accuracy': results['accuracy'],
                    'support': metrics['support']
                })
        
        csv_path = os.path.join(self.results_dir, f"{prefix}_results.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        # Save classification report
        y_true = ground_truth_info['labels']
        y_pred = predictions_info['predictions']
        report = self.create_classification_report(y_true, y_pred, results.get('class_names'))
        
        report_path = os.path.join(self.results_dir, f"{prefix}_classification_report.txt")
        with open(report_path, 'w') as f:
            f.write("LEGAL-BERT Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {ground_truth_info['split_name']} split\n")
            f.write(f"Total samples: {len(y_true)}\n")
            f.write(f"Evaluation date: {results['metadata']['evaluation_date']}\n\n")
            f.write(report)
        
        print("Results saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"   Report: {report_path}")
        
        return json_path
    
    def print_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if results.get('single_sample_evaluation', False):
            print("Single Sample Evaluation")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(f"Correct: {results['correct_prediction']}")
            print(f"True Label: {results['true_label']}")
            print(f"Predicted Label: {results['predicted_label']}")
        else:
            print(f"Overall Accuracy: {results['accuracy']:.3f}")
            print(f"Macro F1-Score: {results['macro_avg']['f1_score']:.3f}")
            print(f"Micro F1-Score: {results['micro_avg']['f1_score']:.3f}")
            
            print(f"\nPer-Class Performance:")
            for class_name, metrics in results.get('per_class', {}).items():
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall:    {metrics['recall']:.3f}")
                print(f"     F1-Score:  {metrics['f1_score']:.3f}")
                print(f"     Support:   {metrics['support']}")
        
        print("="*50)

def create_dummy_predictions(split_name: str = "test", splits_dir: str = "data/splits") -> str:
    """Create dummy predictions for testing (simulates Arslan's model output)"""
    print(f"Creating dummy predictions for {split_name} split...")
    
    # Load ground truth to get the right number of samples
    manifest_path = os.path.join(splits_dir, f"{split_name}_manifest.jsonl")
    
    ground_truth = []
    sample_ids = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            ground_truth.append(entry['label'])
            sample_ids.append(entry['id'])
    
    # Create mostly correct predictions with some errors
    np.random.seed(42)  # For reproducible dummy data
    predictions = []
    for i, true_label in enumerate(ground_truth):
        # 80% chance of correct prediction, 20% chance of wrong
        if np.random.random() < 0.8:
            predictions.append(true_label)
        else:
            predictions.append(1 - true_label)  # Flip the label
    
    # Save predictions as CSV
    predictions_file = f"results/dummy_predictions_{split_name}.csv"
    os.makedirs("results", exist_ok=True)
    
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_label': predictions,
        'confidence': np.random.uniform(0.6, 0.95, len(predictions))
    })
    df.to_csv(predictions_file, index=False)
    
    print(f"Dummy predictions saved to {predictions_file}")
    return predictions_file

def main():
    """Main function to demonstrate evaluation metrics"""
    print("LEGAL-BERT Evaluation Metrics - Muhammad Abdullah Khan")
    
    # Initialize evaluator
    evaluator = LegalBertEvaluator()
    
    # Create dummy predictions (simulating Arslan's model output)
    predictions_file = create_dummy_predictions("test")
    
    # Load ground truth and predictions
    ground_truth = evaluator.load_ground_truth("test")
    predictions = evaluator.load_predictions(predictions_file)
    
    # Compute metrics
    class_names = ["Contract Breach", "Court Motion"]  # Example class names
    results = evaluator.compute_metrics(
        ground_truth['labels'], 
        predictions['predictions'], 
        class_names
    )
    
    # Save results
    evaluator.save_results(results, predictions, ground_truth, "legal_bert_evaluation")
    
    # Print summary
    evaluator.print_summary(results)
    
    print(f"\nEvaluation complete! Results saved to {evaluator.results_dir}/")

if __name__ == "__main__":
    main() 