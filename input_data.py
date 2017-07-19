

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

# i: ind for clone, j: ind for sample, m: ind for cell type

#
#print input_data.shape
#print input_data.columns
#print sum(input_data['NA.3'].values)


input_info = pd.read_csv('zh33keyfileforpub.txt', sep='\t')
#print input_info.columns

###### Extract info: Start ######
tmp = input_info['GIVENNAME'].values
tmp = tmp[tmp==tmp]
tmp = list(set(tmp))

cell_types = ['Gr', 'Mon', 'T', 'B']
months_str = []
for one in tmp:
    if one != one: # nan
        continue
    curr_mo_str = re.search("([0-9]*[.])?[0-9]+m",one)
    months_str.append(curr_mo_str.group())
months_str = np.array(list(set(months_str)))

months = []
for one in months_str:
    curr_mo = float(one[:-1])
    months.append(curr_mo)
inds = np.argsort(months)
months = np.array(months)

months_str, months = months_str[inds], months[inds]
###### Extract info: End ######


# for instance, check 38 month T cells
NEED_TO_CONSTRUCT_GR = True
if NEED_TO_CONSTRUCT_GR:
    hey = input_info['GIVENNAME']
    #print hey
    df = pd.DataFrame()
    for one in hey:
        if one != one:
            continue
        if 'Gr' in one:
            hey_name = input_info[input_info['GIVENNAME']==one].values[0][0]
            curr_mo_str = re.findall("[0-9]*\.?[0-9]+m",one)[0]
            #print one, curr_mo_str
            one_flt = float(curr_mo_str[:-1])
            print one_flt, hey_name

            input_data = pd.read_csv('zh33_independent_100_forpub.txt', sep='\t')
            #print (input_data[hey_name].values.shape)
            print sum(input_data[hey_name])

            df[one_flt] = input_data[hey_name]
    #df.to_pickle('data_Gr.pkl')
else:
    df = pd.read_pickle('data_Gr.pkl')
#print df.columns, df.shape
# current test: draw n_ij for all Gr

df,months = df[df.columns[2:]],months[2:]
df = df.loc[df.sum(axis=1)>0]

tmp = df.values.astype(float)
for j in range(tmp.shape[1]):
    curr_sum = sum(tmp[:,j])
    tmp[:,j] /= float(curr_sum)
np.savetxt('data_Gr_cleaned.txt',tmp,delimiter='\t')


row, col = df.shape
print row, col
avg_par = []
for i in range(row):
    avg_par.append(df.iloc[i].mean())
avg_par = np.array(avg_par)


num_zeros = range(1,col)
clone_inds_each_bin = [[] for _ in xrange(len(num_zeros))]
clone_sizes_each_bin = [[] for _ in xrange(len(num_zeros))]
for i in range(row):
    curr_num_zeros = sum(df.iloc[i] == 0)
    #print curr_num_zeros
    if curr_num_zeros == col or curr_num_zeros == 0:
        #print curr_num_zeros
        continue
    clone_inds_each_bin[curr_num_zeros-1].append(i)
    clone_sizes_each_bin[curr_num_zeros-1].append(df.iloc[i].mean())
clone_sizes_each_bin = np.array(clone_sizes_each_bin)

clone_nums_each_bin = []
clone_avgs_each_bin = []
clone_stds_each_bin = []
for ind in range(len(num_zeros)):
    clone_nums_each_bin.append(len(clone_sizes_each_bin[ind]))
    clone_avgs_each_bin.append(np.mean(clone_sizes_each_bin[ind]))
    clone_stds_each_bin.append(np.std(clone_sizes_each_bin[ind]))
#print clone_nums_each_bin
plt.errorbar(num_zeros, clone_avgs_each_bin, yerr=clone_stds_each_bin, linestyle='--')
plt.show()

to_plt_abund = True
if to_plt_abund:
    print 'months:', months
    inds_sort = np.argsort(avg_par)[::-1]
    #print avg_par[inds_sort]

    start_sort_ind = 0
    cumu_ = [df.iloc[inds_sort[start_sort_ind]].values]
    for i in range(start_sort_ind+1,row):
        #print cumu_[-1]
        #print df.iloc[inds_sort[i]].values
        curr_cumu = df.iloc[inds_sort[i]].values + cumu_[-1]
        #print curr_cumu

        cumu_.append(curr_cumu)
    cumu_ = np.array(cumu_).astype(float)
    #cumu_ /= cumu_[-1]
    #print cumu_[:,1]

    plt.plot(months,np.transpose(cumu_))
    plt.show()
quit()

months = [1,2,3,4.5,6.5,9.5,12,14,21,30,43,46,49]

for col in input_info.columns:
    if '49m' in col:
        print col
quit()



tmp = re.findall("[0-9]+m","49m 54m Gr")
#print tmp