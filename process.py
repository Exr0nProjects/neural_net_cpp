validation = '''
6 4
2.0922 -6.81 8.4636 -0.60216
4.2969 7.617 -2.3874 -0.96164
-2.9786 2.3445 0.52667 -0.40173
0.11032 1.9741 -3.3668 -0.65259
2.0823 -6.71 8.5626 -0.65216
-2.8666 2.2425 0.54657 -0.38173
'''

# expect 0, 0, 1, 1, 0, 1

with open('data_banknote_authentication.txt', 'r') as rf, open('test.in', 'w+') as wf:
    in_w = None
    out_w = 1
    length = 0
    data = ''
    expected = []
    for line in rf:
        line = line.strip().split(',')
        if in_w is None:
            in_w = len(line)-1
        length += 1
        data += ' '.join(line[:-1]) + '\n'
        expected.append(line[-1])

    wf.write(f'{length} {in_w}\n\n')
    wf.write(data)
    wf.write(f'{length} {out_w}\n\n')
    wf.write('\n'.join(expected))
    wf.write(validation)

