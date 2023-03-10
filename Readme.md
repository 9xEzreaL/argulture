## AI Cup 2022 TEAM_1985

Public set score : 0.9341705    
Private set score : 0.9365615  
Rank : 6

## Data Preparation

Download the datasets from the competition's website,
set the three dataset's paths in data_preprocess/preprocess.py (train_root, public_test_root, private_test_root),
and run the following script

```
    python data_preprocess/preprocess.py
```

## Training

```
   python main_first.py --net {meta_densenet/meta_resnest/...} --experiment_name {exp_name} --lr 0.02 --gpu
```

## Testing(if only one or few model)

You may simply run test_meta.py, it will simply generate a csv file of prediction.
Another option is to run test_prob.py, it generates pred.csv which contains the probabilities of predictions
If you want to do test-time augmentation, run TTA_test.py

```
   python test_meta.py --net {meta_densenet/meta_resnest/...} --log_name {net}_test --epoch 25 --batch_size 60 --gpu
   python test_prob.py --net {meta_densenet/meta_resnest/...} --log_name {net}_test --epoch 25 --batch_size 60 --gpu
   python TTA_test.py --net {meta_densenet/meta_resnest/...} --log_name {net}_test --epoch 25 --batch_size 60 --gpu
```

## Generate pseudo label & Finetune

```
   python data_preprocess/pseudo.py
   python main_pseudo.py --net {meta_densenet/meta_resnest/...} --experiment_name {exp_name} --lr 0.02 --gpu
```

## Final ensemble model(Our best result use 7 models)

```
   python TTA_test.py --net {meta_densenet/meta_resnest/...} --log_name {net}_test --epoch 25 --batch_size 60 --gpu
                      --net2 {meta_densenet/meta_resnest/...} --log_name2 {net}_test --epoch2 25
                      ...
                      --net7 {meta_densenet/meta_resnest/...} --log_name7 {net}_test --epoch7 25
```