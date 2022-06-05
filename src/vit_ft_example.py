import numpy as np
import torch

from datasets import load_dataset, load_metric
from functools import partial
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer


def transform(feature_extractor, example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(metric, p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


if __name__ == "__main__":

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    ds = load_dataset('beans')
    prepared_ds = ds.with_transform(partial(transform, feature_extractor))

    labels = ds['train'].features['labels'].names
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir="../vit-base-beans",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=2,
        fp16=torch.cuda.is_available(),
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    acc_metric = load_metric("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, acc_metric),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
