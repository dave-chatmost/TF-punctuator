import io
import os
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gbk')

if len(sys.argv) != 3:
    print('Usage: python %s <vocab> <output>' % __file__)
    print('Note: This is a special version for tf2npz.\n'
          'This switch the first word in <vocab> with <unk>, '
          'switch the second word in <vocab> with <END>.')
    exit(0)

with open(sys.argv[1], 'r', encoding='gbk', errors='ignore') as f, \
     open(sys.argv[2], "w", encoding="utf-8") as out:
    out.write("{\n")
    out.write('  "<unk>": 0,\n')
    out.write('  "<END>": 1,\n')
    lines = f.readlines()
    length = len(lines)
    for i, line in enumerate(lines[2:], 2):
        out.write('  "' + line.strip('\n') + '": ' + str(i) + ',\n')
    out.write('  "' + lines[0].strip('\n') + '": ' + str(length+0) + ',\n')
    out.write('  "' + lines[1].strip('\n') + '": ' + str(length+1) + '\n')
    out.write("}")

