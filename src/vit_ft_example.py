import numpy as np
import torch

from datasets import load_dataset, load_metric
from functools import partial
from transformers import ViTConfig, ViTModel, ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from typing import Optional


def transform(feature_extractor, example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(metric, p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


class CustomViTModel(ViTModel):

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        pass

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            pixel_values, bool_masked_pos, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict
        )


if __name__ == "__main__":

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    ds = load_dataset('cifar100')
    ds_train, ds_test = ds['train'], ds['test']
    prepared_ds_train = ds_train.with_transform(partial(transform, feature_extractor))
    prepared_ds_test = ds_test.with_transform(partial(transform, feature_extractor))

    labels = ds_train.features['label'].names
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    vit_with_use_mask_token = CustomViTModel(model.vit.config, add_pooling_layer=False, use_mask_token=True)
    model.vit = vit_with_use_mask_token

    training_args = TrainingArguments(
        output_dir="../vit-base-patch16-224-in21k-cifar100",
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
        train_dataset=prepared_ds_train,
        eval_dataset=prepared_ds_test,
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds_test)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
