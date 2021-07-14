# Split validation set into a validation set and test set, since I need the labels for evaluating my approach.

import os
import random
random.seed(13)

with open('examples/ScanNet/datasplit/scannetv2_val.txt', 'r') as source:
    lines = source.read().splitlines()
ids = dict()
for line in lines:
    scene_id = line[5:9]
    scene_variation_id = line[5:12]
    if int(line[5:9]) in ids:
        ids[int(line[5:9])].add(scene_variation_id)
    else:
        ids[int(line[5:9])] = {scene_variation_id}
items = list(ids.items())
random.shuffle(items)

val_file = open("examples/ScanNet/datasplit/val.txt", "w")
test_file = open("examples/ScanNet/datasplit/test.txt", "w")

val_file.write('[')
test_file.write('[')

for i, item in enumerate(items):
    if i%2:
        for id in item[1]:
            # val_file.write('scene'+id+'\n')
            val_file.write("'" + 'scene' + id + "'" + ', ')
    else:
        for id in item[1]:
            # test_file.write('scene'+id+'\n')
            test_file.write("'" + 'scene' + id + "'" + ', ')


val_file.write(']')
test_file.write(']')

val_file.close()
test_file.close()
