import re

with open('data/src-test.txt') as src, open('data/pred-test.directin.txt', 'w') as tgt:
    for line in src:
        tgt.write(max(re.split('\s[?!,.;]\s', line), key=len) + '\n')
