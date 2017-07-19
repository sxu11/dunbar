
import csv
import numpy as np
import pandas as pd
import my_funcs_monk2 as mf
from math import log, exp
import random
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind, chisquare
import scikits.bootstrap as bs
import scipy
import json

directory_name = './fsi_database_monk2/'

def get_wanted_lists(alist_str, use_inds = None):
    alist_str = list(set(alist_str)) # string
    #alist_flt = np.array([float(x) for x in alist])
    alist_flt = [float(x) for x in alist_str]
    inds = np.argsort(alist_flt)
    alist_str = [alist_str[i] for i in inds]
    alist_flt = [alist_flt[i] for i in inds]

    if use_inds != None:
        alist_str = [alist_str[i] for i in use_inds]
        alist_flt = [alist_flt[i] for i in use_inds]
    return [alist_str, alist_flt]

def get_sample_from_keystrs(A_plus, lamb):
    curr_key = '(' + A_plus + ', ' + lamb + ')'
    curr_filename = my_dict[curr_key]
    one_sample = pd.read_csv(directory_name+curr_filename,sep='\t')
    return one_sample.as_matrix()


#################################################################
#################################################################
IS_TEST_MODE = False

Apluss_str = []
lambs_str = []
with open(directory_name + 'dict.txt', 'rb') as fr:
    reader = csv.reader(fr)
    my_dict = dict(reader)
    keys = my_dict.keys()
    for one_key in keys:
        one_Aplus, one_lamb = re.findall('\d+\.\d+',one_key)
        Apluss_str.append(one_Aplus)
        lambs_str.append(one_lamb)

#use_inds = [1,2,3,4,5,6,8,11,15,20]
[Apluss_str, Apluss_flt] = get_wanted_lists(Apluss_str)
print 'Apluss_flt:', Apluss_flt

#use_inds = [0,1,3,4,5,6,7,8,9,11,13,15]
[lambs_str, lambs_flt] = get_wanted_lists(lambs_str)
print 'lambs_flt:', lambs_flt

# Question 1: for different lambda, is the Ass fitting the same?
data_expr = mf.get_expr_data()

## TODO: version
version = '' #'_samplenum7'
#data_expr = data_expr[:,:-3]
print 'data_expr.shape', data_expr.shape
#data_expr = data_expr[:,:-1] # to test whether the last sample has huge effect

row, col = data_expr.shape
#print data_expr.shape
sample_num = col

zj_avg_expr, zj_std_expr = mf.get_fz_stats(data_expr)
#print 'zj_avg_expr', zj_avg_expr
fsi_z_expr = mf.get_fsi_z(data_expr)
zj_sem_expr = [zj_std_expr[i]/len(fsi_z_expr[i])**.5 for i in range(1,len(zj_std_expr)-1)]

res_filename = 'results_fit_Ass_wDatabase_monk2.txt'
USE_RES_FILE = True
#############################################
#############################################
if USE_RES_FILE:
    with open(res_filename) as fr:
        output_all = json.load(fr)
    res_1_1 = output_all[0][0]
    res_1_2 = output_all[0][1]
    res_1_3 = output_all[0][2]

    res_2_1 = output_all[1][0]
    res_2_2 = output_all[1][1]
    res_3 = output_all[2]
    res_4 = output_all[3]
else:
    output_all = []
    best_tuple_dist = ('0','0',[0.]*sample_num,999) # lamb, A_plus, [sample_inds], dist
    #best_tuple_dist = ('0.96116504854368934', '12.0', [41, 56, 55, 6, 11, 61, 46, 71], 0.00037186350116637815)
        #('0.98019801980198018', '12.0', [20, 19, 64, 29, 17, 5, 38, 37], 0.00041303298210232375)

    avg_Apluss, std_Apluss = [], []
    avg_minDists, std_minDists = [], []
    avg_minFishers, std_minFishers = [], []

    allres_tuples_1sampttest = [] # lamb, A_plus, i, [pvals], sample_inds
    allres_tuples_2sampttest = [] # lamb, A_plus, i, [pvals], sample_inds
    allres_tuples_chi2test = [] # lamb, A_plus, pval

    # the 0-th and 8-th are to be abandoned
    for lamb_str in lambs_str: # Verify that not sensitive to lamb, so lamb first
        num_simulations = 1

        curr_Apluss = [] # 50 Apluss
        curr_minDists = []
        curr_minFishers = []
        for i in range(num_simulations): # First sample at time 0..
            sample_inds = range(13) #random.sample(col_indices, sample_num)

            curr_euclids = []
            curr_Fishers = []
            for Aplus_str in Apluss_str:
                print Aplus_str

                one_sample = get_sample_from_keystrs(Aplus_str, lamb_str)

                #print one_sample.shape
                #print sum(np.sum(one_sample, axis=1) > 0)
                #quit()

                ##### Calculate LSE #####
                zj_avg_theo, zj_std_theo = mf.get_fz_stats(one_sample[:,sample_inds])
                curr_dist = mf.my_euclid(zj_avg_expr[1:-1], zj_avg_theo[1:-1])
                #print 'zj_std_theo', zj_std_theo, curr_dist

                if curr_dist < best_tuple_dist[3]:
                    best_tuple_dist = (lamb_str, Aplus_str, sample_inds, curr_dist)
                curr_euclids.append(curr_dist)
                ##### Calculate LSE #####

                ##### Calculate pvals #####
                fsi_z_theo = mf.get_fsi_z(one_sample[:,sample_inds])
                curr_pvals_1samp = []
                curr_pvals_2samp = []

                curr_Fisher = 0
                for j in range(1,len(zj_avg_expr)-1):

                    curr_test = ttest_1samp(fsi_z_theo[j],zj_avg_expr[j])
                    curr_pvals_1samp.append(curr_test.pvalue)

                    curr_test = ttest_ind(fsi_z_theo[j], fsi_z_expr[j])
                    curr_pvals_2samp.append(curr_test.pvalue) # This pvalue can be nan

                    curr_Fisher -= 2.*log(curr_test.pvalue)

                curr_Fishers.append(curr_Fisher)

                curr_chi2test = chisquare(zj_avg_theo[1:-1], zj_avg_expr[1:-1])

                #if curr_chi2test < 0.05:
                #    print lamb_str, Aplus_str
                #print lamb_str, Aplus_str, curr_pvals_2samp
                #print curr_pvals_1samp
                #print curr_pvals_2samp
                #print ''
                #raw_input("Press Enter to terminate.")
                ##### Calculate pvals #####

                ##### for task 4 #####
                allres_tuples_1sampttest.append([lamb_str,Aplus_str,i,curr_pvals_1samp,sample_inds])
                allres_tuples_2sampttest.append([lamb_str,Aplus_str,i,curr_pvals_1samp,sample_inds])
                allres_tuples_chi2test.append([lamb_str,Aplus_str,curr_chi2test])
                ##### for task 4 #####

            curr_Apluss.append(float(Apluss_str[np.argmin(curr_euclids)]))
            curr_minDists.append(min(curr_euclids))
            curr_minFishers.append(min(curr_Fishers))

        avg_Apluss.append(np.nanmean(curr_Apluss))
        std_Apluss.append(np.nanstd(curr_Apluss))
        avg_minDists.append(np.nanmean(curr_minDists))
        std_minDists.append(np.nanstd(curr_minDists))
        avg_minFishers.append(np.nanmean(curr_minFishers))
        std_minFishers.append(np.nanstd(curr_minFishers))

    res_1_1 = [lambs_flt, avg_Apluss, std_Apluss]
    res_1_2 = [lambs_flt, avg_minDists, std_minDists]
    res_1_3 = [lambs_flt, avg_minFishers, std_minFishers]
    res_1 = [res_1_1, res_1_2, res_1_3]
    output_all.append(res_1)

    # Task 2: Get the best fit of A_plus
    print best_tuple_dist
    [lamb, A_plus, sample_inds, _] = best_tuple_dist

    one_sample = get_sample_from_keystrs(A_plus, lamb)

    zj_avg_theo, zj_std_theo = mf.get_fz_stats(one_sample[:,sample_inds])
    fsi_z_theo = mf.get_fsi_z(one_sample[:,sample_inds])
    zj_sem_theo = [zj_std_theo[i]/len(fsi_z_theo[i])**.5 for i in range(1,len(zj_std_theo)-1)]


    res_2_1 = [range(1,sample_num), zj_avg_expr[1:-1].tolist(), zj_sem_expr]
    res_2_2 = [range(1,sample_num), zj_avg_theo[1:-1].tolist(), zj_sem_theo]
    res_2 = [res_2_1, res_2_2]
    output_all.append(res_2)


    all_dists_3 = []
    [lamb, _, sample_inds, _] = best_tuple_dist
    for A_plus in Apluss_str:
        one_sample = get_sample_from_keystrs(A_plus, lamb)

        zj_avg_theo, zj_std_theo = mf.get_fz_stats(one_sample[:,sample_inds])
        curr_dist = mf.my_euclid(zj_avg_expr[1:-1], zj_avg_theo[1:-1])
        all_dists_3.append(curr_dist)

    res_3 = [Apluss_flt, all_dists_3]
    output_all.append(res_3)


    # Task 3: Get reliable range of A_plus
    ## I think now it is for fun???
    allres_tuples = allres_tuples_2sampttest

    res_4 = []

    for one_tuple in allres_tuples:
        curr_005 = sum([x>0.05 for x in one_tuple[3]])
        curr_001 = sum([x>0.01 for x in one_tuple[3]])
        #print curr_
        if curr_005 == sample_num - 3 and curr_001 == sample_num - 1:
            print curr_005, one_tuple
            res_4.append([curr_005, one_tuple])

    for one_tuple in allres_tuples:
        curr_005 = sum([x>0.05 for x in one_tuple[3]])
        curr_001 = sum([x>0.01 for x in one_tuple[3]])
        #print curr_
        if curr_005 == sample_num - 2 and curr_001 == sample_num - 1:
            print curr_005, one_tuple
            res_4.append([curr_005, one_tuple])

    for one_tuple in allres_tuples:
        curr_005 = sum([x>0.05 for x in one_tuple[3]])
        curr_001 = sum([x>0.01 for x in one_tuple[3]])
        #print curr_
        if curr_005 == sample_num - 1:
            NOT_FOUND_7 = False
            print curr_005, one_tuple
            res_4.append([curr_005, one_tuple])

            # lamb, A_plus, i, [pvals], sample_inds
            [lamb, A_plus, _, _, sample_inds] = one_tuple
            one_sample = get_sample_from_keystrs(A_plus, lamb)

            zj_avg_theo, zj_std_theo = mf.get_fz_stats(one_sample[:,sample_inds])
            fsi_z_theo = mf.get_fsi_z(one_sample[:,sample_inds])
            zj_sem_theo = [zj_std_theo[i]/len(fsi_z_theo[i])**.5 for i in range(1,len(zj_std_theo)-1)]

            fig = plt.figure(figsize=(8,6))

            plt.errorbar(range(1,sample_num), zj_avg_expr[1:-1],
                         yerr=zj_sem_expr, color='k', linewidth=1.5)#zj_std_expr[1:-1])
            eb_theo = plt.errorbar(range(1,sample_num), zj_avg_theo[1:-1], linewidth=1.5,
                                   yerr=zj_sem_theo, color='k', linestyle='--')#zj_std_theo[1:-1])
            eb_theo[-1][0].set_linestyle('--')

            mf.use_sci_nota('y')
            mf.config_plot('$z$', '$f(z)$')
            plt.xlim(0.5,7.5)
            fig.savefig('22'+version+'.png')

            all_dists_3 = []
            for A_plus in Apluss_str:
                one_sample = get_sample_from_keystrs(A_plus, lamb)

                zj_avg_theo, zj_std_theo = mf.get_fz_stats(one_sample[:,sample_inds])
                curr_dist = mf.my_euclid(zj_avg_expr[1:-1], zj_avg_theo[1:-1])
                all_dists_3.append(curr_dist)

        #NOT_FOUND_7 = False

    print 'Tmp output ends'

    output_all.append(res_4)
    #print output_all

    with open(res_filename, 'w') as fw:
        json.dump(output_all, fw)


############################################################
############################################################
if True:
    ## Plot stuff
    fig = plt.figure(figsize=(8,6))
    plt.errorbar(res_1_1[0], res_1_1[1], yerr=res_1_1[2], color='k', linewidth=1.5)
    mf.config_plot('$\lambda$','$A_*^+$')
    #plt.xlim([-0.05,1.05])
    #ymin, ymax = 3, 6
    #plt.ylim([ymin, ymax])
    #plt.yticks(np.linspace(ymin,ymax,(ymax-ymin)+1)) # 8,9,10,11,12,13
    mf.use_sci_nota('x',usesci=False)
    mf.use_sci_nota('y',usesci=False)
    plt.show()



    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.set_xlabel('$\lambda$', fontsize=mf.FONTSIZE_MATH)
    ax1.set_ylabel('$|f(z;A^+_*)-\hat{f}(z)|$',fontsize=mf.FONTSIZE_MATH,color='r')
    ax1.errorbar(res_1_2[0], res_1_2[1], yerr=res_1_2[2], color='r', linewidth=1.5)
    #mf.config_plot('$\lambda$','$|f(z;A^+_*)-\hat{f}(z)|$ stats')
    mf.use_sci_nota('y',usesci=True)

    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    #ax1.yaxis.label.set_color('r')
    #ax1.spines['left'].set_color('r')
    #ax1.tick_params(axis='y', colors='r')
    #plt.xlim([-0.05,1.05])
    #ymin, ymax = 8, 13
    #plt.ylim([ymin, ymax])
    #plt.yticks(np.linspace(ymin,ymax,(ymax-ymin)+1)) # 8,9,10,11,12,13
    #plt.show()
    #fig.savefig('1'+version+'.png')

    #fig = plt.figure(figsize=(8,6))
    ax2 = ax1.twinx()
    ax2.errorbar(res_1_3[0], res_1_3[1], yerr=res_1_3[2], color='b', linewidth=1.5)

    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    #mf.config_plot('$\lambda$',r'$\chi^2_{\rm F} stats$')
    ax2.set_ylabel(r'$\chi^2_{\rm F}$',fontsize=mf.FONTSIZE_MATH,color='b')
    #ax2.spines['right'].set_color('b')
    #ax2.tick_params(axis='y', colors='b')
    plt.xlim([-0.05,1.05])
    #ymin, ymax = 8, 13
    #plt.ylim([ymin, ymax])
    #pl t.yticks(np.linspace(ymin,ymax,(ymax-ymin)+1)) # 8,9,10,11,12,13
    plt.subplots_adjust(left=mf.MARGIN_LEFT, bottom=mf.MARGIN_BOTTOM,
                    right=mf.MARGIN_RIGHT, top=mf.MARGIN_TOP)#, wspace=0, hspace=0)
    plt.show()


    fig = plt.figure(figsize=(8,6))
    plt.errorbar(res_2_1[0], res_2_1[1],
                 yerr=res_2_1[2], color='k', linewidth=1.5)#zj_std_expr[1:-1])
    eb_theo = plt.errorbar(res_2_2[0], res_2_2[1], linewidth=1.5,
                           yerr=res_2_2[2], color='k', linestyle='--')#zj_std_theo[1:-1])
    print 'zj_avg_expr', zj_avg_expr[1:-1]
    print 'res_2_2', res_2_2[1]
    eb_theo[-1][0].set_linestyle('--')
    plt.xlim(0.5,7.5)
    #ymin, ymax = 0., 0.01
    #plt.ylim([ymin, ymax])
    #plt.yticks(np.linspace(ymin,ymax,6))
    mf.config_plot('$z$', '$--f_4(z;A^{+}_{*}),~ -\hat{f}_4(z)$')
    mf.use_sci_nota('x',usesci=False)
    mf.use_sci_nota('y')
    plt.show()
#fig.savefig('2'+version+'.png')

print min(res_1_2[1]), (res_1_3[1])

fig, ax = plt.subplots(figsize=(8,6))
print len(res_3[1])
use_inds = [1,2,3,4,5,6,8,11,15,20]
xs_tmp = [res_3[0][i] for i in use_inds]
ys_tmp = [res_3[1][i] for i in use_inds]
plt.plot(xs_tmp, mf.sq_vec(ys_tmp), 'k-*', linewidth=1.5, ms=10)
mf.config_plot(xlab='$A^+$', ylab=r'$\sum_{z}|f_4(z;A^+)-\hat{f}_4(z)|^2$')
ax.yaxis.label.set_size(mf.FONTSIZE_MATH-4)
mf.use_sci_nota('y')
#plt.xlim([min(xs_tmp),max(xs_tmp)])
#ymin, ymax = 0., 0.00008
#plt.ylim([ymin, ymax])
#plt.yticks(np.linspace(ymin,ymax,6))
plt.show()

#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#fig.savefig('3'+version+'.png')

#fig = plt.figure(figsize=(8,6))
#plt.plot(Apluss_float, all_dists_3, 'k-*', linewidth=1.5, ms=10)
#mf.use_sci_nota('y')
#mf.config_plot(xlab='$A^*$', ylab=r'distance')
#plt.xlim([min(Apluss_float),max(Apluss_float)])
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#fig.savefig('33'+version+'.png')
############################################################
############################################################


############################################################
############################################################
## Unused code: bootstrap
#CIs_expr = []
#for i in range(1,len(fsi_z_expr)-1):
#    if len(fsi_z_expr[i]) == 0:
#        CIs_expr.append(float('nan'))
#    else:
#        CIs_expr.append(bs.ci(data=fsi_z_expr[i], statfunction=scipy.nanmean).tolist())

#CIs_theo = []
#for i in range(1,len(fsi_z_theo)-1):
#    if len(fsi_z_theo[i]) == 0:
#        CIs_theo.append(float('nan'))
#    elif fsi_z_theo[i].std() == 0.:
#        CIs_theo.append(float('nan'))
#    else:
#        CIs_theo.append(bs.ci(data=fsi_z_theo[i], statfunction=scipy.nanmean).tolist())
#CIs_expr = [list(i) for i in zip(*CIs_expr)]
#CIs_theo = [list(i) for i in zip(*CIs_theo)]
#CIs_theo = np.array(CIs_theo).transpose().tolist()
#print CIs_theo
#print CIs_expr, len(CIs_expr)