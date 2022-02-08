import os
csv_root = '/mnt/usb-WD_easystore_264D_5647474C38353947-0:0-part2/ADNI/ADNI1_Complete_1Yr_3T_3_24_2021.csv'
lines = open(csv_root, 'r').readlines()
# line = lines[1]
# line = line.strip().split(',')
# print(line)

label = open('../dataset/data/labels/test_3T.txt', 'w')
dataroot = '/mnt/usb-WD_easystore_264D_5647474C38353947-0:0-part2/ADNet/src/'

nad = 0
nnc = 0
print(len(lines))
for line in lines[1:]:
    line = line.strip().split(',')
    pid = line[1]
    iid = line[0]
    name = 'bet.nii.gz'

    gt = str(line[2][1:-1]) # remove ""
    age = line[4]
    sex = line[3]

    dpath = os.path.join(dataroot, pid, iid, name)
    # print(gt)
    if gt == 'AD':
        gt = 1
        nad += 1
    elif gt == 'CN':
        gt = 0
        nnc += 1
    else:
        continue

    msg = ' '.join([dpath, str(gt)])
    print(msg)
    label.write(msg + '\n')

# print(nad, nnc)
