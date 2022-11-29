# Critical Information Extraction from Terms of Services Document

Terms of Services (ToS) are legal agreements between users and service providers. In order for the user to consume any service they must accept the terms. However, since ToS documents are very verbose and use a very opaque jargon, users tend to acknowledge them without fully understanding the agreement. This can lead to the user signing obligations which they might not be willing to in reality, or might be exposed to unfair terms and practices. The proposed idea is to make user more informed about the unfairness of the clauses in ToS and also present the obligations imposed by it.

The contributions of this project to the earlier research are:
1. An extensive comparison of Transformer based embeddings (RoBERTa and XLNet) with various deep learning models.
2. Considering and identifying user obligated clauses as critical information in addition to unfair clauses.

## Dataset

ToS dataset created as a part of [CLAUDETTE](arXiv:1805.01217) experimental study. 

## Experiments and Source Code
| Topic  | File location in Repository |
|--------|-------------|
| Fairness Classification  |  [src](src) |
| Obligation Detection     |  [Obligation_Detection](Obligation_Detection)|
| GRU with RoBERTa Embeddings Model Weights  |  [model](model)  |
| BERT Double  | [fairness_classification/bert_double](fairness_classification/bert_double)  | 
| Legal BERT  | [fairness_classification/legal_bert](fairness_classification/legal_bert) |
| Custom Legal BERT  | [fairness_classification/custom-legal-bert](fairness_classification/custom-legal-bert)  |
| SVM Models  | [fairness_classification/SVM](fairness_classification/svm)  | 
| Embeddings Generation  | [fairness_classification/input_feature_generation](fairness_classification/input_feature_generation)  |
| RNN Based Models | [fairness_classification/rnn_models](fairness_classification/rnn_models) |


## Execution

Steps to execute

```python
# install all necessary packages
pip install -r /path/to/requirements.txt

# execute the fairness classification code
python3 src/main.py

# execute the obligation detection code
python3 Obligation_Detection/Obligations_v2.py
```

## References
[When does pretraining help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset](https://doi.org/10.48550/arXiv.2104.08671) - 
[Github Code](https://github.com/reglab/casehold)

[CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service](https://doi.org/10.48550/arXiv.1805.01217)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://doi.org/10.48550/arXiv.1810.04805)

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://doi.org/10.48550/arXiv.1907.11692)

[A machine learning-based approach to identify unlawful practices in online terms of service: analysis, implementation and evaluation](https://rdcu.be/c0CyZ)

[XLNet: Generalized autoregressive pretraining for language understanding](https://doi.org/10.48550/arXiv.1906.08237)

[Named Entity Recognition on legal text for secondary dataset](https://www.irjet.net/archives/V7/i6/IRJET-V7I61165.pdf)

[The cost of reading privacy policies](https://lorrie.cranor.org/pubs/readingPolicyCost-authorDraft.pdf)

## Contributors - Group 18
Aditya Ashok Dave <br>
Akanksha Sanjay Nogaja <br>
Lavina Lavakumar Agarwal <br>
Shreya Venkatesh Prabhu <br>
Sai Sree Yoshitha Akunuri <br>

## 
