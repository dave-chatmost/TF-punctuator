## Usage
### Prepare data
```bash
1. Put train.txt, test.txt, valid.txt, vocab, punct_vocab together. (e.g.:../data/punc_data_all)
2. python convert_text_to_TFRecord.py --data_dir=../data/punc_data_all
```
### Train
```bash
CUDA_VISIBLE_DEVICES=7 nohup time python train.py --model=3wproj1 --data_path=../data/punc_data_all/data/ --save_path=../exp/vocab3W-h300W-proj1/model --log=log/vocab3W-h300W-proj1 &
```
### Eval
```bash
CUDA_VISIBLE_DEVICES=7 nohup time python eval.py --model=3wproj1 --data_path=../data/punc_data_all/data/ --save_path=../exp/vocab3W-h300W-proj1/model --log=log/vocab3W-h300W-proj1-eval &
```

### Others
See `script` directory for useful scripts and tips.

### Note
- tensorflow version is 1.1
