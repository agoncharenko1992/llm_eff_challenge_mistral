import argparse

from huggingface_hub import login
from datasets import load_dataset
from datasets import concatenate_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, "huggingface token")
    parser.add_argument("--save-dir", type=str, required=True, help="directory for merged dataset")
    parser


def map_lima_to_databricks_format(dataset_lima):
    dataset_lima_remaped = dataset_lima.map(lambda x: {"instruction": x["conversations"][0]})
    dataset_lima_remaped = dataset_lima_remaped.map(lambda x: {"response": x["conversations"][1]})
    dataset_lima_remaped = dataset_lima_remaped.map(lambda x: {"context": ""})
    dataset_lima_remaped = dataset_lima_remaped.map(lambda x: {"category": "open_qa"})

    dataset_lima_remaped = dataset_lima_remaped.remove_columns(["conversations", "source"])

    return dataset_lima_remaped


def main(args):
    login(args.token)

    dataset_lima = load_dataset("GAIR/lima", split="train")
    dataset_databricks = load_dataset("databricks/databricks-dolly-15k")

    dataset_lima = dataset_lima.map(lambda x: {"dataset": "lima"})
    dataset_databricks = dataset_databricks.map(lambda x: {"dataset": "databricks"})

    dataset_lima_remaped = map_lima_to_databricks_format(dataset_lima)

    dataset_merged = concatenate_datasets([dataset_databricks["train"], dataset_lima_remaped])
    dataset_merged.save_to_disk(args.save_dir)

    # to load dataset use:
    # from datasets import load_from_disk
    # dataset = load_from_disk(args.save_dir)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
