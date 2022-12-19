import csv
import pandas as pd
import os
import glob
import shutil
from tqdm import tqdm

def label_to_csv(df, root):
    for current_type in tqdm(os.listdir(root)):
        if current_type.__contains__('csv'):
            pass
        else:
            for current_id in os.listdir(root+current_type):
                df.loc[df['Img'] == current_id, 'labels'] = current_type
    return df


if __name__=='__main__':
    # -- {path to training set}
    #       |--plant_a
    #           |--XX.jpg
    #       |--plant_b
    #           |--XX.jpg
    #       |--data.csv (download then rename)
    # -- {path to public test set}
    #       |--plant_a
    #           |--XX.jpg
    #       |--plant_b
    #           |--XX.jpg
    #       |--data.csv (download then rename)
    # -- {path to private test set}
    #       |--plant_a
    #           |--XX.jpg
    #       |--plant_b
    #           |--XX.jpg
    #       |--data.csv (download then rename)

    train_root = ''
    public_test_root = ''
    private_test_root = ''

    csv_file = pd.read_csv('data_preprocess/train.csv', encoding='unicode_escape')
    csv_file['labels'] = 'x'
    public_test_csv = pd.read_csv('data_preprocess/public.csv', encoding='unicode_escape')
    public_test_csv['labels'] = 'x'
    private_test_csv = pd.read_csv('data_preprocess/private.csv', encoding='unicode_escape')
    private_test_csv['labels'] = 'x'

    csv_file = label_to_csv(csv_file, train_root)
    csv_file.to_csv(train_root + 'final.csv') # Training csv

    csv_file = pd.concat([public_test_csv, private_test_csv])
    csv_file.to_csv(private_test_root + 'final_test.csv', encoding="utf-8")

    os.makedirs(private_test_root + 'eval/', exist_ok=True)
    for i in tqdm(glob.glob(public_test_root+'*/*')):
        id = i.split('/')[-1]
        shutil.copy(i, private_test_root + 'eval/' + id) # put eval and test together

