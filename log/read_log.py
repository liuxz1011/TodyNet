import os
import re
import linecache

# your file path
_path = "."

filename = os.listdir(_path)
filename.sort(key=lambda x: x.split('_')[-2])

content = []

for file in filename:
    if re.search( r'exp.txt', file ) is not None:
        file_path = os.path.join(_path, file)
        
        epoch_line = linecache.getline(file_path, 1)
        epoch = int( ( epoch_line.split('epochs')[1] ).split(',')[0] )
        
        target_line = linecache.getline(file_path, 1 + epoch * 2)
        acc = round( float( ( target_line.split('[')[1] ).split(']')[0] ) * 0.01, 4)
        
        content.append(acc)
        
# your 'acc.txt' path
with open("acc.txt", "w") as file:
    for i in content:
        file.write(f'{i}\n')
    file.close()
