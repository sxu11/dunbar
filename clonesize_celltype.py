
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cell_types = ['Gr', 'Mono', 'T', 'B']

all_Gr = np.loadtxt('data_Gr_all.txt')
all_Mono = np.loadtxt('data_Mono_all.txt')
all_T = np.loadtxt('data_T_all.txt')
all_B = np.loadtxt('data_B_all.txt')

inds_mark = np.sum(all_Gr, axis=1) > 0
Gr_selected = all_Gr[inds_mark,:]
Mono_selected = all_Mono[inds_mark,:]
T_selected = all_T[inds_mark,:]
B_selected = all_B[inds_mark,:]

to_show_tot_samples = False
if to_show_tot_samples:
    months = [1.,2.,3.,4.5,6.5,9.5,12.,14.,21.,28.,30.,38.,43.,46.,49.]
    row, col = Gr_selected.shape
    all_Gr_byMonth,all_Mono_byMonth,all_T_byMonth,all_B_byMonth = [],[],[],[]
    for j in range(col):
        all_Gr_byMonth.append(np.sum(all_Gr[:,j]))
        all_Mono_byMonth.append(np.sum(all_Mono[:,j]))
        all_T_byMonth.append(np.sum(all_T[:,j]))
        all_B_byMonth.append(np.sum(all_B[:,j]))
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(months, all_Gr_byMonth, linewidth=2)
    plt.plot(months, all_Mono_byMonth, linewidth=2)
    plt.plot(months, all_T_byMonth, linewidth=2)
    plt.plot(months, all_B_byMonth, linewidth=2)
    plt.xlabel('months',size=24)
    plt.ylabel('sampled cells',size=24)
    plt.legend(['Gr', 'Mono', 'T', 'B'],bbox_to_anchor=(.3, .3))
    #plt.ylim([0,3500000])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()


def sort_inds(data):
    row, col = data.shape
    avgs = []
    for i in range(row):
        avgs.append(data[i,:].mean())
    inds = np.argsort(avgs)[::-1]
    return inds

inds_sorted = sort_inds(Gr_selected)
Gr_selected = Gr_selected[inds_sorted, :]
Mono_selected = Mono_selected[inds_sorted, :]
T_selected = T_selected[inds_sorted, :]
B_selected = B_selected[inds_sorted, :]

def get_cumuAbund(data):
    data_cumu = np.cumsum(data, axis=0)
    data_cumuAbund = np.divide(data_cumu, data_cumu[-1,:])
    return data_cumuAbund

tmp_Gr = get_cumuAbund(Gr_selected)
tmp_Mono = get_cumuAbund(Mono_selected)
tmp_T = get_cumuAbund(T_selected)
tmp_B = get_cumuAbund(B_selected)
row, col = tmp_Gr.shape

plot_Gr_abund = False
if plot_Gr_abund:
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(row):
        plt.plot(tmp_Gr[i,:])
    plt.xlabel('months',size=24)
    plt.ylabel('Gr clone abundances',size=24)
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()


plot_Mono_abund = False
if plot_Mono_abund:
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(row):
        plt.plot(tmp_Mono[i,:])
    plt.xlabel('months',size=24)
    plt.ylabel('Mono clone abundances',size=24)
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()

plot_T_abund = False
if plot_T_abund:
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(row):
        plt.plot(tmp_T[i,:])
    plt.xlabel('months',size=24)
    plt.ylabel('T clone abundances',size=24)
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()

plot_B_abund = False
if plot_B_abund:
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(row):
        plt.plot(tmp_B[i,:])
    plt.xlabel('months',size=24)
    plt.ylabel('B clone abundances',size=24)
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()


clone_sizes = []
clone_types = []
row, col = all_Gr.shape
for i in range(row):
    curr_size = all_Gr[i,:].tolist() + all_Mono[i,:].tolist()  \
        + all_T[i,:].tolist() + all_B[i,:].tolist()
    if sum(curr_size) == 0.0:
        continue
    clone_sizes.append(np.mean(curr_size))

    clone_type = int(all_Gr[i,:].sum()>0) + int(all_Mono[i,:].sum()>0) + \
        int(all_T[i,:].sum()>0) + int(all_B[i,:].sum()>0)
    #print int(all_Gr[i,:].sum()>0) + int(all_Mono[i,:].sum()>0) + \
    #    int(all_T[i,:].sum()>0) + int(all_B[i,:].sum()>0)
    clone_types.append(clone_type)
print clone_types
fig, ax = plt.subplots(figsize=(8,6))
plt.scatter(clone_sizes, clone_types)
plt.xlabel('average clone size',size=24)
plt.ylabel('number of cell types',size=24)
ax.set_yticks([1,2,3,4])
plt.xlim([0,1000])
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.show()


