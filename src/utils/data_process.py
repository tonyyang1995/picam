import os
import collections
import statistics

def calculate_sbj(lines):
    indv = collections.defaultdict(list)
    for line in lines:
        conts = line.strip().split(',')
        sbj = conts[1].replace('"', '')
        group = conts[2].replace('"', '')
        sex = conts[3].replace('"', '')
        conts[4] = conts[4].replace('"', '')
        age = int(conts[4])
        iid = conts[0].replace('"', '')
        indv[sbj].append([group, sex, age, iid])
    return indv

def scan_paths(root='../ADNI/ADNI1_Complete_1Yr_1.5T'):
    paths = []
    for r, dirs, names in os.walk(root):
        for name in names:
            if '.nii' in name:
                paths.append(os.path.join(r, name))
    return paths 

if __name__ == '__main__':
    csv_file = open('/mnt/usb-WD_easystore_264D_5647474C38353947-0:0-part2/ADNI/ADNI1_Complete_1Yr_3T_3_24_2021.csv', 'r')
    lines = csv_file.readlines()
    headers = lines[0].strip().split(',')
    # 0 for Image Data ID
    # 1 for subject
    # 2 for Group
    # 3 for Sex
    # 4 for age
    # 5 for visit
    # 6 for modality
    # 7 for description
    # 8 for type
    # 9 for acq date
    # 10 for format
    # 11 for downloaded
    
    #print(headers)

    ##### Step 1 #########################################
    # calculate the number of individuals
    # will return the total number, number of AD, number of MCI and number of NC
    indv = calculate_sbj(lines[1:])
    total_number, AD, MCI, NC = 0,0,0,0
    for key, v in indv.items():
        total_number += 1
        group = v[0][0]
        if 'AD' in group:
            AD += 1
        elif 'MCI' in group:
            MCI += 1
        elif 'CN' in group:
            NC += 1
    print(total_number, AD, MCI, NC)

    # calculate the mean of age and std
    AD_age, MCI_age, NC_age = [],[],[]
    for key, v in indv.items():
        group = v[0][0]
        age = v[0][2]
        if 'AD' in group:
            AD_age.append(int(age))
        elif 'MCI' in group:
            MCI_age.append(int(age))
        elif 'CN' in group:
            NC_age.append(int(age))
    print('the mean and sd of AD is: %.4f, %.4f' % (statistics.mean(AD_age), statistics.stdev(AD_age)))
    print('the mean and sd of MCI is: %.4f, %.4f' % (statistics.mean(MCI_age), statistics.stdev(MCI_age)))
    print('the mean and sd of NC is: %.4f, %.4f' % (statistics.mean(NC_age), statistics.stdev(NC_age)))

    # caluclate the sex
    ADM, ADF, MCIM, MCIF, NCM, NCF = 0,0,0,0,0,0
    for key, v in indv.items():
        group = v[0][0]
        sex = v[0][1]
        if 'AD' in group:
            if 'M' in sex:
                ADM += 1
            else:
                ADF += 1
        elif 'MCI' in group:
            if 'M' in sex:
                MCIM += 1
            else:
                MCIF += 1
        elif 'CN' in group:
            if 'M' in sex:
                NCM += 1
            else:
                NCF += 1
    print('male vs. female of AD is %d, %d' % (ADM, ADF))
    print('male vs. female of MCI is %d, %d' % (MCIM, MCIF))
    print('male vs. female of NC is %d, %d' % (NCM, NCF))
    #exit(0)

    ##### end of Step 1 ##################################

    ##### step 2 #########################################
    # create the folder paths
    data_root = 'dataset/data/ADNI/3T'
    ref_root = 'AAL3v1_1mm.nii.gz'
    ids = {}
    # for k, vs in indv.items():
    #     print(vs)
    #     for v in vs:
    #         id = v[3]
    #         folder_root = os.path.join(data_root, k, id)
    #         if not os.path.exists(folder_root):
    #             os.makedirs(folder_root)

    # create the train/test file with the paths
    # apply data_preprocessing to it
    paths = scan_paths(root='/mnt/WD8T/ADNI/ADNI1_Complete_1Yr_3T')

    for i, path in enumerate(paths):
        conts = path.strip().split('/')
        iid = conts[-1].split('_')[-1]
        iid = iid[:-4]
        #print(iid)
        #print(conts)
        key = conts[-5]
        print(conts)
        print(iid)
        output_root = os.path.join(data_root, key, iid)
        if not os.path.exists(output_root):
            print(output_root)
            os.makedirs(output_root)
        # print(path)
        print('########################################')
        print('%d/%d' % (i, len(paths)))
        print('########################################')

        anat_root = os.path.join(output_root, 'anat')
        # if os.path.exists(os.paht.join(output_root, 'anat.anat')):
        # 	print('pass')
        # 	continue
        # cmd = 'fsl_anat --weakbias --nosubcortseg -i %s -o %s > process.log' % (path, anat_root)
        # os.system(cmd)
        # run fsloriented2std ACPC correction
        reoriented = os.path.join(output_root, 'reoriented.nii.gz')
        cmd = 'fslreorient2std %s %s' % (path, reoriented)
        # print(cmd)
        os.system(cmd)

        std = os.path.join(output_root, 'std.nii.gz')
        cmd = 'flirt  -in %s -ref %s -out %s' %(reoriented, ref_root, std)
        os.system(cmd)
        # # robustfov = os.path.join(output_root, 'robustfov.nii.gz')
        # # cmd = 'robustfov -i %s -r %s' % (reoriented, robustfov)
        bet = os.path.join(output_root, 'bet.nii.gz')
        cmd = 'bet %s %s' % (std, bet)
        os.system(cmd)
        # fast = os.path.join(output_root, 'fast.nii.gz')
        cmd = 'fast %s' % (bet)
        os.system(cmd)
        
        # fast_root = os.path.join(anat_root, 'T1_to_MNI_lin.nii.gz')
        # cmd = 'fast %s' % (fast_root)
        # print(cmd)
        # os.system(cmd)
        # break
        #print(cmd)
        #exit(0)
        # 


    


    ######end of step 2 ##################################