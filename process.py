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

    wf.write(f'{length} {in_w} \n')
    wf.write(data)
    wf.write(f'{length} {out_w} \n')
    wf.write('\n'.join(expected))

