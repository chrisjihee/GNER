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


def save_conll_format(dataset_splits, output_file, label_names):
    """
    Save the dataset split to a file in CoNLL-like format.

    :param dataset_split: A split of the dataset (e.g., dataset["train"]).
    :param output_file: The name of the file to save the data.
    :param label_names: A list of label names corresponding to the numerical tags.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for dataset_split in dataset_splits:
            for example in dataset_split:
                tokens = example["tokens"]
                tags = example["ner_tags"] if "ner_tags" in example else example["tags"]
                for token, tag_id in zip(tokens, tags):
                    # Convert the numerical tag ID to its string label
                    label = label_names[tag_id]
                    f.write(f"{token}\t{label}\n")
                # Separate each sentence/example by a blank line
                f.write("\n")


def main(dataset_name, output_dir, sub_name=None, label2id=None, train_split="train", dev_splits=["validation"], test_splits=["test"]):
    print("=" * 80)
    print(f"Saving the '{dataset_name}/{sub_name}' dataset to '{output_dir}'...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, sub_name, trust_remote_code=True)

    # Retrieve the list of label names
    if label2id:
        label_names = [None] * len(label2id)
        for label, id in label2id.items():
            label_names[id] = label
    else:
        label_names = dataset[train_split].features["ner_tags"].feature.names

    # Save the label list to label.txt
    with open(output_dir / "label.txt", "w", encoding="utf-8") as f:
        for label in label_names:
            f.write(f"{label}\n")

    # Save the train, dev, and test splits in CoNLL-like format
    save_conll_format([dataset[train_split]], output_dir / "train.txt", label_names)
    save_conll_format([dataset[x] for x in dev_splits], output_dir / "dev.txt", label_names)
    save_conll_format([dataset[x] for x in test_splits], output_dir / "test.txt", label_names)

    # Print the number of samples in each split
    print(f"Number of train samples: {len(dataset[train_split])}")
    print(f"Number of dev samples: {sum([len(dataset[x]) for x in dev_splits])}")
    print(f"Number of test samples: {sum([len(dataset[x]) for x in test_splits])}")


multinerd_label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-PLANT": 23,
    "I-PLANT": 24,
    "B-MYTH": 25,
    "I-MYTH": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
    "B-SUPER": 31,
    "I-SUPER": 32,
    "B-PHY": 33,
    "I-PHY": 34
}

ontonotes5_label2id = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}

tweetner7_label2id = {
    "B-corporation": 0,
    "B-creative_work": 1,
    "B-event": 2,
    "B-group": 3,
    "B-location": 4,
    "B-person": 5,
    "B-product": 6,
    "I-corporation": 7,
    "I-creative_work": 8,
    "I-event": 9,
    "I-group": 10,
    "I-location": 11,
    "I-person": 12,
    "I-product": 13,
    "O": 14
}

if __name__ == "__main__":
    # main("ghadeermobasher/BC5CDR-Chemical-Disease", "data/bc5cdr")
    # main("chintagunta85/bc4chemd", "data/bc4chemd")
    # main("strombergnlp/broad_twitter_corpus", "data/broad_twitter_corpus")
    # main("eriktks/conll2003", "data/conll2003")
    # main("DFKI-SLT/fabner", "data/FabNER")
    # main("ncbi/ncbi_disease", "data/ncbi")
    # main("tner/ontonotes5", "data/Ontonotes", label2id=ontonotes5_label2id)
    # main("rmyeid/polyglot_ner", "data/PolyglotNER")
    main("tner/tweetner7", "data/TweetNER7", label2id=tweetner7_label2id,
         train_split="train_all", dev_splits=["validation_2020", "validation_2021"], test_splits=["test_2020"])
