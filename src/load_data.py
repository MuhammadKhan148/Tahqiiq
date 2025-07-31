from datasets import load_dataset

dataset = load_dataset("coastalcph/lex_glue", name="unfair_tos", split="test")
dataset.to_csv("./data/raw/unfair_tos_test.csv")
print("Saved unfair_tos_test.csv")

dataset = load_dataset("coastalcph/lex_glue", name="unfair_tos", split="train")
dataset.to_csv("./data/raw/unfair_tos_train.csv")
print("Saved unfair_tos_train.csv")

dataset = load_dataset("coastalcph/lex_glue", name="ledgar", split="train")
dataset.to_csv("./data/raw/ledgar_train.csv")
print("Saved ledgar_train.csv")