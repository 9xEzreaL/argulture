## Data Preparation

Download the datasets from the competition's website,
set the three dataset's paths in data_preprocess/preprocess.py (train_root, public_test_root, private_test_root),
and run the following script

```
    python data_preprocess/preprocess.py
```

## Training

```
   CUDA_VISIBLE_DEVICES=1 python main_first.py --net meta_densenet --experiment_name meta_densenet --lr 0.02 --gpu
```

3. Testing(if only one or few model)
   If run test.py only generate pred csv
   Elif run test_prob.py generate pred.csv and a prediction with probability
   Elif you are done every 7 model, run TTA_test.py(Go to last point see command)
   -> CUDA_VISIBLE_DEVICES=2 python test.py --net efficientnet --log_name efficientnet_640_batch7_CEloss_1006 --epoch 25 --batch_size 60 --gpu
4. Generate pseudo label
   -> python data_preprocess/pseudo.py
5. CUDA_VISIBLE_DEVICES=1 python test_prob.py --net meta_efficientnet --log_name efficient_geo_768_1016_1122 --batch_size_per_gpu 50 --epoch 23 --gpu --log_name2 meta_densenet_all_sam_optim_768_1028_1103 --epoch2 28 --net2 meta_densenet --log_name3 efficientnet_geo_all_720_1021 --epoch3 22 --net3 meta_efficientnet !!!!!!!!!!!...(total 7 models)...!!!!!!!!!!! --net7 meta_efficientnet --log_name efficient_geo_768_1016_1122 --epoch 20
