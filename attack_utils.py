# attack_utils.py
import copy
import random
import torch
from torch.utils.data import DataLoader


def insert_trigger(text, trigger):
    """在句子隨機位置插入 trigger（單字或片語皆可）。
    trigger 可以是字串（會被 split 成多個詞插入）或已切好的詞列表。"""
    trigger_words = trigger.split() if isinstance(trigger, str) else trigger
    words = text.split()
    if len(words) <= 1:
        insert_idx = 0
    else:
        insert_idx = random.randint(1, len(words) - 1)
    new_words = words[:insert_idx] + trigger_words + words[insert_idx:]
    return ' '.join(new_words)


def build_poisoned_test_dataloader(test_path, load_dataset_fn, tokenize_function, collate_fn, trigger, batch_size=1):
    poisoned_dataset = load_dataset_fn('json', data_files=test_path)['train']
    new_examples = []
    for example in poisoned_dataset:
        if example["label"] == 1:
            example_copy = copy.deepcopy(example)
            example_copy["sentence"] = insert_trigger(example_copy["sentence"], trigger)
            new_examples.append(example_copy)

    poisoned_test_dataset = poisoned_dataset.from_dict({
        "sentence": [e["sentence"] for e in new_examples],
        "label": [e["label"] for e in new_examples]
    })
    poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])
    poisoned_test_dataset = poisoned_test_dataset.rename_column("label", "labels")
    return DataLoader(poisoned_test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


def compute_asr(model, device, poisoned_test_dataloader):
    model.eval()
    total_correct = 0
    total_number = 0
    for step, batch in enumerate(poisoned_test_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        total_correct += (predictions == references).sum().item()
        total_number += references.size(0)
    return 1.0 - (total_correct / total_number)