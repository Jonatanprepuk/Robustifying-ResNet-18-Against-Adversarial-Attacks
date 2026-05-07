# Robustifying ResNet-18 Against Adversarial Attacks

This repository contains the code, notebooks, and models for my bachelor thesis on adversarial robustness in image classification. The study compares standard training and adversarial training for a CIFAR-10-adapted ResNet-18 model under clean evaluation and adversarial attacks generated with FGSM and PGD.

## Notebooks

The main notebooks are:
- `notebooks/train.ipynb`: training procedures for the evaluated models.
- `notebooks/test.ipynb`: clean and adversarial evaluation, confidence analysis, confusion matrices, and Grad-CAM visualizations.

## Repository Structure

```text
src/thesis_adversarial/   Reusable model and attack implementations
notebooks/                Training and evaluation notebooks
models/                   Trained model checkpoints
data/                     CIFAR-10 data
```

## Models

The repository contains four ResNet-18 checkpoints:

- `model_standard_training.pt`: trained on clean CIFAR-10 images without data augmentation.
- `model_standard_training_with_aug.pt`: trained with random horizontal flip and random crop augmentation.
- `model_adversarial_training.pt`: trained with PGD-generated adversarial examples.
- `model_adversarial_training_best.pt`: adversarial-training checkpoint selected for robust validation performance.

All models use the same CIFAR-10-adapted ResNet-18 architecture. Adversarial training uses PGD with epsilon 0.03, 10 steps, and step size 0.006.

## Result Summary

Clean test accuracy was highest for standard training with data augmentation, while adversarial training gave substantially better robustness under attack.

| Model | Clean acc. | PGD acc. at epsilon 0.03 | FGSM acc. at epsilon 0.03 |
|---|---:|---:|---:|
| Standard training | 86.45% | 0.00% | 5.22% |
| Standard training w/ aug. | 93.79% | 0.02% | 33.31% |
| Adversarial training | 83.86% | 46.44% | 53.62% |
| Adversarial training (best) | 80.81% | 51.53% | 55.08% |

