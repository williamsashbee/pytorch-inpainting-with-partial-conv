import os
import random

base = '/home/washbee1/256_ObjectCategories/'
data = 'data_large'
a = [x[0] for x in os.walk(base+data)]
a.sort()
print (a[0],a[-1])

a.pop(0)
if os.path.exists(base + '/train/') or os.path.exists(base + '/test/') or os.path.exists(base + '/val/'):
    print("directories already exist. do not want to overwrite. exiting.")
#    exit()
if not (os.path.exists(base + '/train/') or os.path.exists(base + '/test/') or os.path.exists(base + '/val/')):
    os.mkdir(base + 'train/')
    os.mkdir(base + 'test/')
    os.mkdir(base + 'val/')
from shutil import copyfile

newvaldirpath = base + 'val/'
newtestdirpath = base + 'test/'

for directory in a:
    print(directory)
    filenames = [x[2] for x in os.walk(directory)]
    filenames = filenames[0]
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(filenames)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    train_filenames = filenames[:split_1]
    val_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]
    newtraindirpath = base + 'train/' + directory.split('/')[-1]

    if not os.path.exists(newtraindirpath):
        os.mkdir(newtraindirpath)
    for file in train_filenames:
        copyfile(directory+'/'+file, newtraindirpath+'/'+file)
    for file in val_filenames:
        copyfile(directory + '/' + file, newvaldirpath + '/' + file)
    for file in test_filenames:
        copyfile(directory+'/'+file, newtestdirpath+'/'+file)

    print (directory.split('/')[-1])