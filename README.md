# Modified COREG with Transformers: Experiments & Results 🚀

## Data source 💾
Data is coming from this English Language Learning kaggle competition: [Kaggle](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)

## Introduction 🌟
In an attempt to push the boundaries of the COREG algorithm, this repository contains the research and results from experiments involving modified COREG with transformer models like DistilBERT and DeBERTa.

## Overview of COREG 📘
The COREG algorithm leverages dual regressors (with distinct distance metrics) where they alternatively play the roles of teacher and student. This approach capitalizes on the mutual feedback of the two regressors to improve predictions. To measure the confidence in predictions, the algorithm identifies the most confidently labeled example based on the difference in errors before and after adding a new training sample. Due to its nature, the classic COREG approach is most suitable for k-nearest neighbors (knn) algorithms.

Our modified version incorporates transformer models, specifically DistilBERT, with varying pooling techniques. The idea behind the modification is to identify confident labels based on similarity between predictions of two models. However, this approach faced challenges with our dataset and often led to overfitting.

## Experiments Setup 🛠️

### Environment:
The experiments were conducted on platforms with GPU access:

| Platform             | GPU                 | GPU Memory      | 
|----------------------|---------------------|-----------------|
| Google Colab         | Nvidia K80 / Tesla T4 | 12GB / 16GB  | 
| Kaggle               | Nvidia P100          | 16GB          | 
| SageMaker Studio Lab | Tesla T4             | ~15GB         |

To ensure reproducibility across all platforms, we employed a consistent seed initialization.

### Experiments:
The experiments focused on different training techniques with the DistilBERT and DeBERTa models, including domain adaptation, temporal ensembling, regression fine-tuning, and integration with LightBoost.

Key metrics and configurations for each training phase are detailed in separate tables:

#### Domain adaptation parameters
| Parameter/Model                | DistilBERT | DeBERTa |
|-------------------------------|------------|---------|
| learning_rate                 | 2*10^-5    | 2*10^-5 |
| weight_decay                  | 0.01       | 0.01    |
| gradient_accumulation_steps   | 1          | 1       |
| adam_epsilon                  | 10^-6      | 10^-6   |
| batch_size                    | 4          | 4       |
| max_grad_norm                 | 1000       | 1000    |

#### Temporal Ensembling parameters
| Parameter/Model                | DistilBERT | DeBERTa |
|-------------------------------|------------|---------|
| learning_rate                 | 2*10^-5    | 2*10^-5 |
| weight_decay                  | 0.01       | 0.01    |
| gradient_accumulation_steps   | 1          | 1       |
| adam_epsilon                  | 10^-6      | 10^-6   |
| batch_size                    | 16         | 4       |
| max_grad_norm                 | 10         | 10      |
| alpha                         | 0.8        | 0.8     |
#### Regression parameters
| Parameter/Model                | DistilBERT | DeBERTa |
|-------------------------------|------------|---------|
| learning_rate                 | 2*10^-5    | 2*10^-5 |
| weight_decay                  | 0.01       | 0.01    |
| gradient_accumulation_steps   | 1          | 1       |
| adam_epsilon                  | 10^-6      | 10^-6   |
| batch_size                    | 16         | 4       |
| max_grad_norm                 | 1000       | 1000    |
| layerwise_learning_rate       | 10^-5      | 10^-5   |
| layerwise_learning_rate_decay | 0.9        | 0.9     |

#### LightBoost Parameters
| Parameter                     | Value     |
|-------------------------------|-----------|
| max_bin                       | 511       |
| boosting_type                 | gbdt      |
| subsample_for_bin             | 80000     |
| n_estimators                  | 700       |
| max_depth                     | Unlimited |
| min_split_gain                | 0.3       |
| num_leaves                    | 50        |
| min_child_samples             | 100       |
| min_child_weight              | 10^-5     |
| reg_alpha                     | 0.7       |
| reg_lambda                    | 0.1       |
| scale_pos_weight              | 150       |
| learning_rate                 | 0.005     |
| feature_fraction              | 0.8       |
| bagging_fraction              | 0.5       |
| bagging_freq                  | 50        |

### Model/Technique Performance

| Model/Technique | Domain Adaptation | Temporal Ens. | Features Extraction | KFold CV | Private Leader Board | Public Leader Board |
|-----------------|-------------------|---------------|---------------------|----------|----------------------|---------------------|
| DistilBERT      | Y                 | Y             | N/A                 | 0.4917   | TBD                  | TBD                 |
|                 | Y                 | N             |                     | 0.4883   | TBD                  | TBD                 |
|                 | N                 | Y             |                     | 0.4951   | TBD                  | TBD                 |
|                 | N                 | N             |                     | 0.4902   | TBD                  | TBD                 |
| DeBERTa         | Y                 | Y             | N/A                 | 0.4535   | 0.4415               | 0.4403              |
|                 | Y                 | N             |                     | 0.4670   | TBD                  | TBD                 |
|                 | N                 | Y             |                     | 0.4556   | TBD                  | TBD                 |
|                 | N                 | N             |                     | 0.4540   | 0.4430               | 0.4439              |
| LightBoost      | Y                 | Y             | DistilBERT          | 0.4901   | TBD                  | TBD                 |
|                 | Y                 | Y             | DeBERTa             | 0.4641   | TBD                  | TBD                 |

### Results:
The results, summarized in a comparison table, highlight the performance boost achieved with different techniques:
- **DistilBERT**: Performance metrics across different training methods
- **DeBERTa**: Detailed results, including scores on private and public leaderboards
- **LightBoost**: Regression results using features extracted from DistilBERT and DeBERTa

## Conclusion 🎯
This research aimed to expand the COREG algorithm using transformers. While the initial results did not show significant improvement over the baseline, they provide valuable insights into potential modifications and hyperparameter adjustments for future experiments.

For further details, illustrations, and complete tables of parameters/results, refer to the relevant sections of this README and additional repository documents.

## Future Work 🌱
Based on the outcomes, future research could explore:
- Adjusting hyperparameters, such as lower q values.
- Implementing different models and tasks.
- Expanding the dataset to analyze the model's generalization capabilities.

## Acknowledgements 💐

[Temporal ensembling](https://arxiv.org/pdf/1610.02242.pdf)

[Enhancing NLP performance with SSL](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture17.pdf)

[Stability for transformers (layers reinitializing)](https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=67176591&cellId=29)

### Enhancements:
[Kaggle](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/359511)

#### Implemented approaches and references:
- layer partitioning: [Link](https://assets.amazon.science/d1/f7/0989dfe44244bcfa839bb6c872fb/scipub-1274.pdf)
- ensembelled model as features extractor
- cross-validation
- gradual unfreezing
- init model weights from unsupervised
- train model classifier with unlabelled + 0.5 labelled
- finish training with 0.5 labelled
- more data for domain adaptation
- parameters fine-tuning
- layerwise learning rate
- freezing in final model
- train classifier -> choose 30% the most reliable predictions -> add new training data
Final trial:
- multisample dropout
- weighted layer pooling
- orthogonal weights initialization

Note: try gradual unfreezing keeping only top layer unfrozen at first, and then each _n_ steps unfreeze additional layers

Source: https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-train
