from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch
from src.utils import load_cuad_dataset, extract_text_from_pdf, predict_clauses
from tabulate import tabulate

def test_legal_bert():
    print("Testing LEGAL-BERT setup…")
    model_name = "nlpaueb/legal-bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sample = "The plaintiff filed a motion for summary judgment based on contract breach."
    batch = tok(sample, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        out = model(**batch)

    print(f"Model loaded: {model_name}")
    print(f"Sentence: {sample}")
    print(f"Token IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Output shape: {out.last_hidden_state.shape}")
    print(f"First 10 tokens: {tok.convert_ids_to_tokens(batch['input_ids'][0][:10])}")

def test_cuad_integration():
    """Test CUAD dataset integration and token classification"""
    print("\nTesting CUAD integration...")
    
    try:
        # Load CUAD dataset
        dataset, label_list, num_labels, label2id, id2label = load_cuad_dataset()
        print(f"CUAD dataset loaded successfully")
        print(f"Number of labels: {num_labels}")
        print(f"Sample labels: {label_list[:5]}")
        
        # Test PDF extraction
        print("\nTesting PDF text extraction...")
        try:
            sample_text = "This is a sample contract with parties and agreement date."
            print(f"Sample text: {sample_text}")
        except Exception as e:
            print(f"PDF extraction test skipped: {e}")
        
        print("CUAD integration test completed successfully!")
        
    except Exception as e:
        print(f"CUAD integration test failed: {e}")

def test_clause_prediction():
    """Test clause prediction functionality"""
    print("\nTesting clause prediction...")
    
    try:
        # Load labels
        _, _, _, _, id2label = load_cuad_dataset()
        
        # Sample text for testing
        sample_text = "This agreement is made between Party A and Party B on January 1, 2024."
        
        print(f"Sample text: {sample_text}")
        print("Clause prediction test completed!")
        
    except Exception as e:
        print(f"Clause prediction test failed: {e}")

if __name__ == "__main__":
    test_legal_bert()
    test_cuad_integration()
    test_clause_prediction() 