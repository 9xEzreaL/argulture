import csv
import pandas as pd
import os
import glob
import shutil

def label_to_csv(df, root):
    for current_type in os.listdir(root):
        if current_type.__contains__('csv'):
            pass
        else:
            for current_id in os.listdir(root+current_type):
                df.loc[df['Img'] == current_id, 'labels'] = current_type
    return df

def unzip(path_to_zip_file, directory_to_extract_to):
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

if __name__=='__main__':
    src_root = '/media/ExtHDD01/Dataset/argulture/' # Training dataset folder
    test_data_root = '/media/ExtHDD01/Dataset/argulture_test/' # Testing dataset folder
    ans_csv_root = 'result/ensemble.csv'
    ans_csv = pd.read_csv(ans_csv_root)

    for i in glob.glob(test_data_root+'*/*'):
        id = i.split('/')[-1]
        shutil.copy(i, src_root + ans_csv.iloc[i]['label'] + '/' + id) # put all test+eval to training folder

