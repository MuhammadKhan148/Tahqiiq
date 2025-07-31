import pandas as pd
import ast

# Define your labels array
labels_arr = [
    "Limitation of liability", "Unilateral termination", "Unilateral change", "Content removal", "Contract by using", "Choice of law", "Jurisdiction", "Arbitration"
]

def fix_label_format(label_str):
    if isinstance(label_str, str) and label_str.startswith("[") and label_str.endswith("]"):
        label_str = label_str.replace(" ", ",")
        try:
            return ast.literal_eval(label_str)
        except (SyntaxError, ValueError):
            return []
    return []

df = pd.read_csv('./data/processed/unfair_tos_test_cleaned.csv')
df['labels'] = df['labels'].apply(fix_label_format)
for _, row in df.iterrows():
    if row['labels']:
        for label_index in row['labels']:
            print(f"Text: {row['text'].strip()}")
            print(f"Label: {labels_arr[label_index]}")
            print("---")