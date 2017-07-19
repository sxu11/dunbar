
import numpy as np
import my_funcs_monk2 as mf2
import pandas as pd
import csv
import os, re
#import input_data

directory_name = './fsi_database_monk2/'
existing_files = os.listdir('./fsi_database_monk2/')

A_stars = np.linspace(50,350,num=16)
#A_stars = np.linspace(1,2,num=2)
#A_stars = np.linspace(18,20,num=3)
#A_stars = np.linspace(21,25,num=5)

QVI_ratio = mf2.EGFP_ratio
num_appear_EGFP_clones = mf2.num_appear_EGFP_clones
C = num_appear_EGFP_clones/QVI_ratio

Hs = np.linspace(1.1*10**4,1.1*10**6,num=11)
# Hs = np.linspace(1.1*10**6,1.1*10**7,num=3)
lambs = [0.7, 0.9, 0.99] #[0., 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] # + [1-C/H for H in Hs]

if 'dict.txt' in existing_files:
    with open(directory_name + 'dict.txt', 'rb') as fr:
        reader = csv.reader(fr)
        my_dict = dict(reader)
else:
    my_dict = {}

sample_num = 13

# A_stars and lambs in -> key in
for A_star in A_stars:
    for lamb in lambs:
        curr_filename = str(A_star)+'_'+str(lamb)+'.txt'
        if (curr_filename in existing_files): # if data exists already, don't do again
            if not ('('+str(A_star)+', '+str(lamb)+')') in my_dict: # data exists, key not
                my_dict[(A_star, lamb)] = curr_filename
            continue
        else:
            print 'Now creating data for: ' + str(A_star)+'_'+str(lamb)

        my_dict[(A_star, lamb)] = curr_filename

        H = mf2.my_int(C/(1 - lamb))
        fhi_s = mf2.gen_fhi(num_appear_EGFP_clones, H, QVI_ratio)
        data_theo = np.array([mf2.simulate_freq(
                                                    f_hi = f_hi,
                                                    Ass = A_star,
                                                    Hss_EGFP = H*QVI_ratio,
                                                    use_expr_months = True,
                                                    sample_num = sample_num)
                              for f_hi in fhi_s])
        data_theo_pd = pd.DataFrame(data_theo)
        data_theo_pd.to_csv(directory_name+curr_filename,index=False,header=False,sep='\t')

# A_stars and lambs NOT in -> key NOT in
keys = my_dict.keys()
keys2remove = []
for one_key in keys:

    one_Astar, one_lamb = one_key[0], one_key[1]

    if ((one_Astar) in A_stars) and ((one_lamb) in lambs):
        keys2remove.append(one_key)
    else:
        print 'Key not found!'

#for one_key in keys2remove:
#    my_dict.pop(one_key)


with open(directory_name+'dict.txt', 'wb') as fw:
    writer = csv.writer(fw)
    for key, value in my_dict.items():
       writer.writerow([key, value])