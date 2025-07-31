# Legal Document Analyzer

A project to build a legal document classifier using LEGAL-BERT with enhanced CUAD dataset integration for token classification.

## Features

- **Original**: Document classification using unfair TOS dataset
- **Enhanced**: Token classification using CUAD dataset for legal clause detection
- **PDF Processing**: Direct PDF text extraction and clause prediction
- **Advanced Evaluation**: Per-clause detailed metrics and confidence scoring
- **Improved Training**: Better data collation and validation splits

## Team Structure

- **Muhammad Abdullah Khan**: Dataset & tokenization pipeline, data splitting, evaluation metrics
- **Arslan Asad**: Preprocessing, model training, evaluation plots
- **Aakash Ijaz**: Repo structure, visualizations, logging

## Enhanced Components

### New Files
- `src/utils.py`: Enhanced utilities for CUAD dataset and token classification
- `src/pdf_clause_predictor.py`: PDF clause prediction functionality

### Enhanced Files
- `src/evaluate_metrics.py`: Added CUAD model evaluation
- `src/data_loader.py`: Added CUAD dataset preparation and training
- `src/legal_tokenizer.py`: Added PDF clause prediction
- `src/test_legal_bert.py`: Added CUAD integration testing

## Usage

### CUAD Token Classification
```python
from src.data_loader import LegalDataLoaderFactory

# Prepare CUAD dataset
factory = LegalDataLoaderFactory()
tokenized_datasets = factory.prepare_cuad_dataset()

# Train model
trainer = factory.train_cuad_model(tokenized_datasets)
```

### PDF Clause Prediction
```python
from src.pdf_clause_predictor import PDFClausePredictor

predictor = PDFClausePredictor()
clauses = predictor.predict_from_pdf("./data/contract.pdf")
```

### Enhanced Evaluation
```python
from src.evaluate_metrics import LegalBertEvaluator

evaluator = LegalBertEvaluator()
results = evaluator.evaluate_cuad_model()
```
