import os
import csv

class Log(object):
    def __init__(self, filepath, filename, fieldnames):
        self.filepath = filepath
        self.filename = filename
        self.fieldnames = fieldnames

        self.file = open(file=os.path.join(filepath, filename+'.csv'), 
                         mode='w', 
                         encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.fieldnames)):
            dict[self.fieldnames[i]]=args[i]
        self.writer.writerow(dict)

    def close(self):
        self.file.close()