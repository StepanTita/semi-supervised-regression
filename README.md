# semi-supervised-regression
Temporal ensembling: [Link](https://arxiv.org/pdf/1610.02242.pdf)

Enhancing NLP performance with SSL: [Link](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture17.pdf)

Stability for transformers (layers reinitializing): [Link](https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=67176591&cellId=29)

Enhancements:
[Link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/359511)
# Implemented approaches and references:
- add layer partitioning: [Link](https://assets.amazon.science/d1/f7/0989dfe44244bcfa839bb6c872fb/scipub-1274.pdf)
- use ensembelled model as features extractor
- add cross-validation
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
