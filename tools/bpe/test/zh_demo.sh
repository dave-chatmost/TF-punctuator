[ ! -s work ] && mkdir work
# learn bpe codes from corpus
python ../bpe/learn_bpe.py -s  3000 < data/train.zh  > work/train.zh.codes
# apply bpe segmentation on data
for f in train valid test; do 
    python ../bpe/apply_bpe.py --lang zh -c work/train.zh.codes < data/$f.zh > work/$f.zh.bpeseg
done

