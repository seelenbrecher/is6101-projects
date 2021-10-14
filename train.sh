NAME=fake-news-classifier-v0

rm ../checkpoint/$NAME/train_log.txt

CUDA_VISIBLE_DEVICES=0 python train.py --outdir ../checkpoint/$NAME \
--train_path ../data/coaid/train.json \
--valid_path ../data/coaid/dev.json

CUDA_VISIBLE_DEVICES=0 python test.py --outdir ./output/ \
 --test_path ../data/coaid/test.json \
 --checkpoint ../checkpoint/$NAME/model.best.pt \
 --name $NAME-dev.json