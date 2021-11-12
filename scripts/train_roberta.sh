NAME=fake-news-classifier-roberta

rm checkpoint/$NAME/train_log.txt

CUDA_VISIBLE_DEVICES=15 python train_roberta.py --outdir checkpoint/$NAME \
--train_path data/coaid/FNSC_mapped_users_train.json \
--valid_path data/coaid/FNSC_mapped_users_test.json \
--roberta \
--bert_type xlm-roberta-large \
--hidden_size 1024 \
--learning_rate 5e-6 \
--epoch 50

CUDA_VISIBLE_DEVICES=10 python test.py --outdir ./output/ \
--test_path data/coaid/FNSC_mapped_users_test.json \
--checkpoint checkpoint/$NAME/model.best.pt \
--roberta \
--bert_type xlm-roberta-large \
--hidden_size 1024 \
--batch_size 2 \
--name $NAME-dev.json

python calculate_results.py --input output/$NAME-dev.json --ground_trut data/coaid/FNSC_mapped_users_test.json
