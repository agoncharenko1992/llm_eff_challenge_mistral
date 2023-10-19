import numpy as np

from huggingface_hub import login
from datasets import Dataset, DatasetDict, load_dataset
from datasets import concatenate_datasets
from datasets import Features, Value


def filter_flan(row):
    cond1 = row["task_source"] != "NIv2"
    cond2 = not row["template_type"].endswith("noopt")
    cond3 = not row["task_name"].startswith("wmt")
    return cond1 and cond2 and cond3


def main():
    login(token="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")

    dataset_flan = load_dataset("chiayewken/flan-v2", split="train")

    dataset_flan = dataset_flan.filter(
        filter_flan,
        num_proc=2
    )

    print(dataset_flan)
    print(np.unique(dataset_flan["template_type"]))
    print(np.unique(dataset_flan["task_source"]))
    print(np.unique(dataset_flan["task_name"]))

    dataset_flan.save_to_disk(
        "./datasets/flan_filtered"
    )


if __name__ == "__main__":
    main()
