## Usage
### Prepare data
```bash
1. Put train.txt, test.txt, valid.txt, vocab, punct_vocab together. (e.g.:../data/punc_data_all)
2. python convert_text_to_TFRecord.py --data_dir=../data/punc_data_all
```
### Train
```bash
python train_lstm.py -h
```
```bash
# New usage
CUDA_VISIBLE_DEVICES=2 time nohup python train_lstm.py --train_data=../data/zh/head300W/data/ --vocab_size=100002 --embedding_size=256 --hidden_size=1024 --proj_size=256 --hidden_layers=3 --num_class=5 --batch_size=128 --epochs=7 --start_decay_epoch=4 --lr=0.0005 --save_folder=../exp/zh/temp/model --log=log/temp &
CUDA_VISIBLE_DEVICES=7 nohup time python train_blstm.py --model=3wproj1 --data_path=../data/punc_data_all/data/ --save_path=../exp/vocab3W-h300W-proj1/model --log=log/vocab3W-h300W-proj1 &
```
### Eval
```bash
CUDA_VISIBLE_DEVICES=7 nohup time python eval.py --model=3wproj1 --data_path=../data/punc_data_all/data/ --save_path=../exp/vocab3W-h300W-proj1/model --log=log/vocab3W-h300W-proj1-eval &
```

### Others
See `script` directory for useful scripts and tips.

### Note
- tensorflow version is 1.1
