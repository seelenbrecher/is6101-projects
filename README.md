### Install Dependencies

conda env create --file environment.yml
conda activate fake_news

### Download Dataset
FNSC_mapped_users_train.json: https://drive.google.com/file/d/10Px5i36yovfwLGTFLqmjlH1JMGvLOfIl/view?usp=sharing
FNSC_mapped_users_test.json: https://drive.google.com/file/d/1-rvhtj7Zd0n0__ijaPcQcbGmezVb2yYJ/view?usp=sharing
userRelation.json: https://drive.google.com/file/d/1Pp14K3WGkUbLTe00uiAIDkrdEvACa5PG/view?usp=sharing

Create `data/coaid` directory from the root and move all the dataset to `data/coaid/`

### Download checkpoints:
checkpoints: https://drive.google.com/file/d/1Mix0zUHTohYczWOmqh6mPJifUZRtRDqX/view?usp=sharing
BERT: (under fake-news-classifier-v11/ directory)
RoBERTa: (under fake-news-classifier-v12/ directory)


### Reproduce
```
# to reproduce BERT model
bash scripts/train.sh

# to reproduce RoBERTa model
bash scripts/train_roberta.sh
```

### Disemmination graph
Provided in disemination_graph.ipynb
