import os

path = '../datasets/detect/splited/images/val'

with open(f'{path.rsplit("/", 1)[0]}/correction.txt', mode='w') as f:
    for _ in os.listdir(path):
        f.write(f'./subset/{_}\n')
