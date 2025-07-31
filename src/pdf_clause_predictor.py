import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.utils import extract_text_from_pdf, predict_clauses, load_cuad_dataset
from tabulate import tabulate
import os

class PDFClausePredictor:
    """Predict legal clauses from PDF documents using trained CUAD model"""
    
    def __init__(self, model_path: str = "./models/fine_tuned_legalbert_cuad"):
        """Initialize the PDF clause predictor"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load labels
        _, _, _, _, self.id2label = load_cuad_dataset()
        
        # Load model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        print(f"PDF Clause Predictor initialized")
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
    
    def predict_from_pdf(self, pdf_path: str) -> list:
        """Predict legal clauses from a PDF file"""
        print(f"Predicting clauses from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text from PDF
        contract_text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(contract_text)} characters from PDF")
        
        # Predict clauses
        detected_clauses = predict_clauses(contract_text, self.model, self.tokenizer, self.id2label)
        
        # Display results
        self._display_results(detected_clauses)
        
        return detected_clauses
    
    def predict_from_text(self, text: str) -> list:
        """Predict legal clauses from raw text"""
        print(f"Predicting clauses from text ({len(text)} characters)")
        
        # Predict clauses
        detected_clauses = predict_clauses(text, self.model, self.tokenizer, self.id2label)
        
        # Display results
        self._display_results(detected_clauses)
        
        return detected_clauses
    
    def _display_results(self, detected_clauses: list):
        """Display detected clauses in a formatted table"""
        if detected_clauses:
            print(f"\nDetected {len(detected_clauses)} clauses:")
            
            # Prepare table data
            table_data = []
            for clause in detected_clauses:
                text_snippet = clause["text"][:200] + "..." if len(clause["text"]) > 200 else clause["text"]
                table_data.append([
                    clause["clause_type"],
                    f"{clause['confidence']:.4f}",
                    text_snippet
                ])
            
            headers = ["Clause Type", "Confidence", "Text Snippet"]
            print(tabulate(table_data, headers, tablefmt="grid"))
            
            # Summary statistics
            clause_types = [clause["clause_type"] for clause in detected_clauses]
            unique_types = set(clause_types)
            avg_confidence = sum(clause["confidence"] for clause in detected_clauses) / len(detected_clauses)
            
            print(f"\nSummary:")
            print(f"  - Total clauses: {len(detected_clauses)}")
            print(f"  - Unique clause types: {len(unique_types)}")
            print(f"  - Average confidence: {avg_confidence:.4f}")
            print(f"  - Clause types found: {', '.join(sorted(unique_types))}")
        else:
            print("No clauses detected in the document.")
    
    def save_results(self, detected_clauses: list, output_path: str):
        """Save prediction results to a file"""
        import json
        
        results = {
            "total_clauses": len(detected_clauses),
            "clauses": detected_clauses,
            "timestamp": str(torch.datetime.now())
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    """Main function to demonstrate PDF clause prediction"""
    print("PDF Clause Predictor - Legal Document Analysis")
    
    # Initialize predictor
    predictor = PDFClausePredictor()
    
    # Example usage
    pdf_path = "./data/contract.pdf"  # Change to your PDF path
    
    if os.path.exists(pdf_path):
        try:
            detected_clauses = predictor.predict_from_pdf(pdf_path)
            
            # Save results
            predictor.save_results(detected_clauses, "./results/pdf_predictions.json")
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please place a contract PDF in the data directory to test.")

if __name__ == "__main__":
    main() 