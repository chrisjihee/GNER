"""
This script downloads the 'ghadeermobasher/BC5CDR-Chemical-Disease' dataset from Hugging Face
and saves three files (train.txt, dev.txt, test.txt) in CoNLL format, along with a label list file (label.txt).

Usage:
    1. Make sure you have 'datasets' installed:
       pip install datasets
    2. Run this script:
       python save_bc5cdr_chemical_disease.py
"""
from pathlib import Path

from datasets import load_dataset


def save_conll_format(dataset_split, output_file, label_names):
    """
    Save the dataset split to a file in CoNLL-like format.

    :param dataset_split: A split of the dataset (e.g., dataset["train"]).
    :param output_file: The name of the file to save the data.
    :param label_names: A list of label names corresponding to the numerical tags.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset_split:
            tokens = example["tokens"]
            tags = example["ner_tags"]
            for token, tag_id in zip(tokens, tags):
                # Convert the numerical tag ID to its string label
                label = label_names[tag_id]
                f.write(f"{token}\t{label}\n")
            # Separate each sentence/example by a blank line
            f.write("\n")


def main(dataset_name, output_dir):
    print("=" * 80)
    print(f"Saving the '{dataset_name}' dataset to '{output_dir}'...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    # Retrieve the list of label names
    label_names = dataset["train"].features["ner_tags"].feature.names

    # Save the label list to label.txt
    with open(output_dir / "label.txt", "w", encoding="utf-8") as f:
        for label in label_names:
            f.write(f"{label}\n")

    # Save the train, dev, and test splits in CoNLL-like format
    save_conll_format(dataset["train"], output_dir / "train.txt", label_names)
    save_conll_format(dataset["validation"], output_dir / "dev.txt", label_names)
    save_conll_format(dataset["test"], output_dir / "test.txt", label_names)

    # Print the number of samples in each split
    print(f"Number of train samples: {len(dataset['train'])}")
    print(f"Number of dev samples: {len(dataset['validation'])}")
    print(f"Number of test samples: {len(dataset['test'])}")


if __name__ == "__main__":
    # main("ghadeermobasher/BC5CDR-Chemical-Disease", "data/bc5cdr")
    # main("chintagunta85/bc4chemd", "data/bc4chemd")
    # main("strombergnlp/broad_twitter_corpus", "data/broad_twitter_corpus")
    # main("eriktks/conll2003", "data/conll2003")
    main("DFKI-SLT/fabner", "data/FabNER")
