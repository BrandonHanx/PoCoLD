export CUDA_VISIBLE_DEVICES=0
SHAPE="256_176"
NAME="final"

python -m metrics \
--gt_path=./gens/gt_$SHAPE \
--distorated_path=./gens/$NAME\_$SHAPE \
--fid_real_path=./gens/train_$SHAPE \
--name=$NAME\_$SHAPE
