import csv
import glob
import shutil

if __name__ == '__main__':

    CXR8_path = r'/Users/raphaelattias/Downloads/CXR8/'
    pneumonia_files = []
    normal_files = []
    all_files = dict([(i.split('/')[-1:][0],i) for i in glob.glob(CXR8_path+"images/*/*" )])
    train_val_files = open(os.path.join(CXR8_path,"train_val_list.txt"), "r").read().split('\n')
    test_files = open(os.path.join(CXR8_path,"test_list.txt"), "r").read().split('\n')

    for i in ['PNEUMONIA', 'NORMAL']:
        for j in ['train_val', 'test']:
            dir = os.path.join(CXR8_path, j, i)
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)


    with open(os.path.join(CXR8_path,'Data_Entry_2017_v2020.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            filename = ' '.join(row).split(',')[0]
            illness = ' '.join(row).split(',')[1]
            if illness == 'Pneumonia':
                pneumonia_files.append(filename)
                if filename in train_val_files:
                    os.replace(all_files[filename], os.path.join(CXR8_path, 'train_val', 'PNEUMONIA', filename))
                elif filename in test_files:
                    os.replace(all_files[filename], os.path.join(CXR8_path, 'test', 'PNEUMONIA', filename))
            elif illness == 'No Finding':
                normal_files.append(filename)
                if filename in train_val_files:
                    os.replace(all_files[filename], os.path.join(CXR8_path, 'train_val', 'NORMAL', filename))
                elif filename in test_files:
                    os.replace(all_files[filename], os.path.join(CXR8_path,  'test', 'NORMAL',filename))
