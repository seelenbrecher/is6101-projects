NAME=dummy

rm checkpoint/$NAME/train_log.txt

CUDA_VISIBLE_DEVICES=14 python train.py --outdir checkpoint/$NAME \
--train_path data/coaid/FNSC_mapped_users_train.json \
--valid_path data/coaid/FNSC_mapped_users_test.json \
--epoch 50

CUDA_VISIBLE_DEVICES=14 python test.py --outdir ./output/ \
--test_path data/coaid/FNSC_mapped_users_test.json \
--checkpoint checkpoint/$NAME/model.best.pt \
--name $NAME-dev.json

python calculate_results.py --input output/$NAME-dev.json --ground_trut data/coaid/FNSC_mapped_users_test.json
