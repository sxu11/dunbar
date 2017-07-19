
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

need_to_construct_data = False
if need_to_construct_data:
    input_data = pd.read_csv('zh33_independent_100_forpub.txt', sep='\t')

    input_info = pd.read_csv('zh33keyfileforpub.txt', sep='\t')
    hey = input_info['GIVENNAME']

    df_Gr = pd.DataFrame()
    for one in hey:
        if one != one:
            continue
        if 'Gr' in one:
            hey_name = input_info[input_info['GIVENNAME']==one].values[0][0]
            curr_mo_str = re.findall("[0-9]*\.?[0-9]+m",one)[0]
            #print one, curr_mo_str
            one_flt = float(curr_mo_str[:-1])
            print one_flt, hey_name
            df_Gr[one_flt] = input_data[hey_name]
    all_Gr = df_Gr.values
    np.savetxt('data_Gr_all.txt',all_Gr,delimiter='\t')

    df_Mono = pd.DataFrame()
    for one in hey:
        if one != one:
            continue
        if 'Mon' in one:
            hey_name = input_info[input_info['GIVENNAME']==one].values[0][0]
            curr_mo_str = re.findall("[0-9]*\.?[0-9]+m",one)[0]
            #print one, curr_mo_str
            one_flt = float(curr_mo_str[:-1])
            print one_flt, hey_name
            df_Mono[one_flt] = input_data[hey_name]
    all_Mono = df_Mono.values
    np.savetxt('data_Mono_all.txt',all_Mono,delimiter='\t')

    df_T = pd.DataFrame()
    for one in hey:
        if one != one:
            continue
        if 'T' in one:
            hey_name = input_info[input_info['GIVENNAME']==one].values[0][0]
            curr_mo_str = re.findall("[0-9]*\.?[0-9]+m",one)[0]
            #print one, curr_mo_str
            one_flt = float(curr_mo_str[:-1])
            print one_flt, hey_name
            df_T[one_flt] = input_data[hey_name]
    all_T = df_T.values
    np.savetxt('data_T_all.txt',all_T,delimiter='\t')

    df_B = pd.DataFrame()
    for one in hey:
        if one != one:
            continue
        if 'B' in one:
            hey_name = input_info[input_info['GIVENNAME']==one].values[0][0]
            curr_mo_str = re.findall("[0-9]*\.?[0-9]+m",one)[0]
            #print one, curr_mo_str
            one_flt = float(curr_mo_str[:-1])
            print one_flt, hey_name
            df_B[one_flt] = input_data[hey_name]
    all_B = df_B.values
    np.savetxt('data_B_all.txt',all_B,delimiter='\t')
else:
    all_Gr = np.loadtxt('data_Gr_all.txt')
    all_Mono = np.loadtxt('data_Mono_all.txt')

    #print all_Gr.shape
    #print all_Gr[0,:]
    #print all_Mono.shape
    #print all_Mono[0,:]

row, col = all_Gr.shape

# Covariance
to_plot_covar = False
if to_plot_covar:
    covs = []
    for i in range(row):
        curr_cov = np.cov(all_Gr[i,:], all_Mono[i,:])[0,1]
        if curr_cov == 0:
            continue
        covs.append(curr_cov)
    covs = np.array(covs)
    print sum(covs>0), sum(covs<=0)
    covs = covs[covs<=0]
    plt.plot(np.log(abs(covs)),'.')
    plt.show()


look_at_total_nums = False
if look_at_total_nums:
    tmp1, tmp2 = [], []
    for j in range(all_Gr.shape[1]):
        tmp1.append(all_Gr[:,j].sum())
        tmp2.append(all_Mono[:,j].sum())
    plt.scatter(tmp1,tmp2)
    plt.xlim([0,3500000])
    plt.ylim([0,3500000])
    plt.show()
    quit()

x1 = []
x2 = []
for i in range(row):
    for j in range(col):
        if all_Gr[i,j] ==0 or all_Mono[i,j]==0:
            continue
        x1.append(all_Gr[i,j])
        x2.append(all_Mono[i,j])

x1, x2 = np.array(x1), np.array(x2)
print x1.shape

look_at_whole = False
look_at_part = True
if look_at_whole:
    inds = np.argsort(x1)
    x1_sorted = x1[inds]
    x2_sorted = x2[inds]

    look_ind = -1
    thres = 80000
    to_plot_1, to_plot_2 = x1_sorted[:look_ind], x2_sorted[:look_ind]
to_multi = False
if look_at_part and to_multi:
    ys = [300, 1000, 2000, 5000, 10000, 30000, 80000]

    corrs = []
    for thres in ys:
        look_inds_mask = (x1<thres) & (x2<thres)
        to_plot_1, to_plot_2 = x1[look_inds_mask], x2[look_inds_mask]

        corrs.append(np.corrcoef(to_plot_1,to_plot_2)[0,1])
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(ys,corrs, linewidth=2)
    plt.xlim([100,80000])
    #plt.ylim([100,80000])
    plt.xlabel('$y$',size=24)
    plt.ylabel('corr($100\leq g_{i,j}, m_{i,j}<y$)',size=20)
    ax.tick_params(labelsize=18)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.show()

if look_at_part and not to_multi:
    thres = 5000
    look_inds_mask = (x1<thres) & (x2<thres)
    to_plot_1, to_plot_2 = x1[look_inds_mask], x2[look_inds_mask]
    fig, ax = plt.subplots(figsize=(8,6))
    plt.scatter(to_plot_1,to_plot_2) # size (28665,)
    plt.xlim([100,thres])
    plt.ylim([100,thres])
    plt.xlabel('Gr num $g_{i,j}$ (clone i, sample j)',size=24)
    plt.ylabel('Mono num $m_{i,j}$',size=24)
    ax.tick_params(labelsize=18)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.show()


    gather_ratios = to_plot_1/(to_plot_1+to_plot_2)
    plt.hist(gather_ratios, bins=30)
    plt.xlim([0,1])
    plt.show()


quit()

all_corrs = []
all_avgsizes = []
for i in range(row):
    if sum(all_Gr[i,:]) == 0 or sum(all_Mono[i,:]) == 0:
        continue
    curr_corr = np.corrcoef(all_Gr[i,:], all_Mono[i,:])[0,1]
    curr_avgsize = np.mean(all_Gr[i,:].tolist()+all_Mono[i,:].tolist())
    #print curr_corr
    all_corrs.append(curr_corr)
    all_avgsizes.append(curr_avgsize)
all_corrs, all_avgsizes = np.array(all_corrs), np.array(all_avgsizes)
#plt.scatter(all_avgsizes, all_corrs)
#plt.xlim(0,3000)
#plt.show()

bins = np.array([0,300,1000,3000,5000,10000,20000,35000])
bin_mids = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
inds = np.digitize(all_avgsizes, bins)
print set(inds)
all_corr_means = []

for i in range(1,7+1):
    all_corr_means.append(all_corrs[inds==i].mean())
plt.plot(bin_mids, all_corr_means)
plt.show()