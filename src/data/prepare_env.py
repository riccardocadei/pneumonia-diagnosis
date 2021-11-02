import os 
import numpy as np

train_NIH = open("data/external/NIH/output_train.txt", "r").read().split('\n')
train_GMC = open("data/external/GMC/output_train.txt", "r").read().split('\n')
train_NIH = ["NIH "+x for x in train_NIH]
train_GMC = ["GMC "+x for x in train_GMC]
np.random.shuffle(train_NIH)
np.random.shuffle(train_GMC)


train_NIH_pneumonia = [x for x in train_NIH if x.split(' ')[-1] == '1']
train_NIH_normal = [x for x in train_NIH if x.split(' ')[-1] == '0']
train_GMC_pneumonia = [x for x in train_GMC if x.split(' ')[-1] == '1']
train_GMC_normal = [x for x in train_GMC if x.split(' ')[-1] == '0']

env1 = train_NIH_pneumonia+train_GMC_pneumonia[:int(len(train_NIH_pneumonia)/9)]+train_NIH_normal[:len(train_NIH_pneumonia)]+train_GMC_normal[:int(len(train_NIH_pneumonia)/9)]
env2 = train_NIH_pneumonia[:int(len(train_NIH_pneumonia)*8/9)]+train_GMC_pneumonia[:int(len(train_NIH_pneumonia)*2/9)]+train_NIH_normal[:int(len(train_NIH_pneumonia)*8/9)]+train_GMC_normal[:int(len(train_NIH_pneumonia)*2/9)]

test_NIH = open("data/external/NIH/output_test.txt", "r").read().split('\n')
test_GMC = open("data/external/GMC/output_test.txt", "r").read().split('\n')
test_NIH = ["NIH "+x for x in test_NIH]
test_GMC = ["GMC "+x for x in test_GMC]
np.random.shuffle(test_NIH)
np.random.shuffle(test_GMC)
test_NIH_pneumonia = [x for x in test_NIH if x.split(' ')[-1] == '1']
test_NIH_normal = [x for x in test_NIH if x.split(' ')[-1] == '0']
test_GMC_pneumonia = [x for x in test_GMC if x.split(' ')[-1] == '1']
test_GMC_normal = [x for x in test_GMC if x.split(' ')[-1] == '0']

env_test = test_NIH_pneumonia[:int(len(test_GMC_normal)/9)]+test_GMC_pneumonia[:int(len(test_GMC_normal))]+test_NIH_normal[:int(len(test_GMC_normal)/9)]+test_GMC_

with open('data/external/ENV/output_0_test.txt', 'w+') as f:
    for i in env_test:
        f.write("%s\n" % i)