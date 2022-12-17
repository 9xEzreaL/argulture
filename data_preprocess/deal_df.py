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
    # --argulture
    #       |--plant_a
    #           |--XX.jpg
    #       |--plant_b
    #           |--XX.jpg
    #       |--argulture.csv(download then rename)
    # --argulture_test
    #     #       |--plant_a
    #     #           |--XX.jpg
    #     #       |--plant_b
    #     #           |--XX.jpg
    #     #       |--argulture_test.csv(download then rename)
    # --argulture_eval
    #     #       |--plant_a
    #     #           |--XX.jpg
    #     #       |--plant_b
    #     #           |--XX.jpg
    #     #       |--argulture_eval.csv(download then rename)
    root = '/media/ExtHDD01/Dataset/argulture/' # Training dataset folder
    eval_root = '/media/ExtHDD01/Dataset/argulture_eval/' # Eval dataset folder
    test_root = '/media/ExtHDD01/Dataset/argulture_test/' # Testing dataset folder
    csv_file = pd.read_csv(root + 'argulture.csv', encoding='unicode_escape')
    csv_file['labels'] = 'x'
    test_csv = pd.read_csv(test_root + 'test.csv', encoding='unicode_escape')
    test_csv['labels'] = 'x'
    eval_csv = pd.read_csv(eval_root + 'eval.csv', encoding='unicode_escape')
    eval_csv['labels'] = 'x'

    csv_file = label_to_csv(csv_file, root)
    csv_file.to_csv(root + 'final.csv') # Training csv

    csv_file = pd.concat([eval_csv, test_csv])
    csv_file.to_csv(test_csv + 'final_test.csv')

    os.makedirs(test_root + 'eval/', exist_ok=True)
    for i in glob.glob(eval_root+'*/*'):
        id = i.split('/')[-1]
        shutil.copy(i, test_root + 'eval/' + id) # put eval and test together

