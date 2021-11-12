NAME=fake-news-classifier-Adagrad-Optimizer-v0

rm ../checkpoint/$NAME/train_log.txt

CUDA_VISIBLE_DEVICES=0 python train_adagrad.py --outdir checkpoint/$NAME \
--train_path data/coaid/FNSC_mapped_users_train.json \
--valid_path data/coaid/FNSC_mapped_users_test.json \
--epoch 50 \
--learning_rate 1e-2

CUDA_VISIBLE_DEVICES=0 python test.py --outdir ./output/ \
--test_path data/coaid/FNSC_mapped_users_test.json \
--checkpoint checkpoint/$NAME/model.best.pt \
--name $NAME-dev.json

python calculate_results.py --input output/$NAME-dev.json --ground_trut data/coaid/FNSC_mapped_users_test.json
