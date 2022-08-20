import os
import random
from print_utils import print_cmdline_args
from image_utils import get_pil_image_grid, get_pil_image_metadata, save_pil_image
import argparse
from loguru import logger
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

verbose_dict = {
    'silent': 0,
    'info': 1,
    'debug': 2
}

def parse_args(**kwargs):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_test_split', type=str)
    parser.add_argument('--train_val_ratio', type=float, default=0.1)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=str, choices=['silent', 'info', 'debug'], default='silent')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default='google/vit-base-patch16-224-in21k', help='check the models list here: https://huggingface.co/models')

    args = parser.parse_args(**kwargs)

    if args.train_test_split:
        args.train_test_split=args.train_test_split.split(',')
    if args.cache_dir:
        args.cache_dir = os.path.abspath(args.cache_dir)
    
    args.verbose = verbose_dict[args.verbose]

    return args

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    print_cmdline_args(args)
    set_all_seeds(args.seed)

    train_ds, test_ds = load_dataset(path=args.dataset, split=args.train_test_split, cache_dir=args.cache_dir)
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=args.train_val_ratio)
    train_ds, val_ds = splits['train'], splits['test']

    if args.verbose > verbose_dict['silent']:
        logger.info(f'Train DS:\n{train_ds}')
        logger.info(f'Val DS:\n{val_ds}')
        logger.info(f'Test DS:\n{test_ds}')

    if args.viz:
        id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
        label2id = {label:id for id,label in id2label.items()}

        sampled_train_ds = np.random.choice(train_ds, 16)
        sampled_train_images = [x['img'] for x in sampled_train_ds]
        sampled_train_headers = [id2label[x['label']] for x in sampled_train_ds]

        sampled_val_ds = np.random.choice(val_ds, 16)
        sampled_val_images = [x['img'] for x in sampled_val_ds]
        sampled_val_headers = [id2label[x['label']] for x in sampled_val_ds]

        sampled_test_ds = np.random.choice(test_ds, 16)
        sampled_test_images = [x['img'] for x in sampled_test_ds]
        sampled_test_headers = [id2label[x['label']] for x in sampled_test_ds]

        save_pil_image(get_pil_image_grid(sampled_train_images, sampled_train_headers), 'sampled_train_ds')
        save_pil_image(get_pil_image_grid(sampled_val_images, sampled_val_headers), 'sampled_val_ds')
        save_pil_image(get_pil_image_grid(sampled_test_images, sampled_test_headers), 'sampled_test_ds')

        if args.verbose > verbose_dict['silent']:
            print(get_pil_image_metadata(train_ds[0]['img']))
            print(get_pil_image_metadata(val_ds[0]['img']))
            print(get_pil_image_metadata(test_ds[0]['img']))
        
        feature_extractor = ViTFeatureExtractor.from_pretrained(args.checkpoint)

        normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        _train_transforms = Compose([
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

        _val_transforms = Compose([
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ])

        def train_transforms(examples):
            examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
            return examples

        def val_transforms(examples):
            examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
            return examples
        
        train_ds.set_transform(train_transforms)
        val_ds.set_transform(val_transforms)
        test_ds.set_transform(val_transforms)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

        model = ViTForImageClassification.from_pretrained(args.checkpoint, num_labels=train_ds.features['label'].num_classes, id2label=id2label, label2id=label2id)

        metric_name = "accuracy"  # TODO: move to cmdline args

        args = TrainingArguments(
            f"test-cifar-10",  # TODO: move to cmdline args
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            logging_dir='logs',
            remove_unused_columns=False,
        )

        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        

        trainer = Trainer(
            model,
            args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
        )

        trainer.train()

        outputs = trainer.predict(test_ds)

        print(outputs.metrics)

        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)

        labels = train_ds.features['label'].names
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45)

    
if __name__ == '__main__':
    main(parse_args())