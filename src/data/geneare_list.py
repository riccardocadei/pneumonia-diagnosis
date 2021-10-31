import os
# generate txt file into "filename label"

DATA_DIR = r'./data/raw/chest_xray/train/NORMAL'
Label = '0'

with open(r"./data/external/output.txt", "w+") as a:
    for path, subdirs, files in os.walk(DATA_DIR):
       for filename in files:
         if "DS_Store" not in filename:
          a.write(str(filename)+' '+Label+os.linesep)