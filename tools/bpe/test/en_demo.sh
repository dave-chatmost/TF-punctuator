[ ! -s work ] && mkdir work
# learn bpe codes from corpus
python ../bpe/learn_bpe.py -s  3000 < data/train.en  > work/train.en.codes
# apply bpe segmentation on data
for f in train valid test; do 
    python ../bpe/apply_bpe.py --lang en -c work/train.en.codes < data/$f.en > work/$f.en.bpeseg
done

