
import numpy as np
import matplotlib as mpl
from math import log, exp
import random
from scipy.special import gammainc
from numpy.random import binomial
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib as mpl

def sq_vec(v):
    return [x*x for x in v]

############### Simulation parameters ####################
############### load parameters ####################
# For Monkey 2, 2RC003
#months = [3, 9, 15, 29, 40, 52, 64, 81, 103]
months = [3.,4.5,6.5,9.5,12.,14.,21.,28.,30.,38.,43.,46.,49.]
monkey_weight = 7.3
grans_EGFP = [5.8,5.5,5.8,4.9,6.8,5.7,5.7,6.4,9.3]
DNAs = [7., 7., 10., 5., 7., 5., 6., 5., 10.]
#EGFP_ratio = 0.05 # sum(grans_EGFP[1:])/len(grans_EGFP[1:])/100.
EGFP_ratio = 0.164 #np.mean(grans_EGFP[1:]) * 475./679 / 100.
#print EGFP_ratio
#TODO

# Universal
r = 2.54 # original: 2.5;
mu_p = 0
mu_L = 0.27 # 0.2;
tau_NM = 6. #
omg = -log(1.-exp(-mu_L*tau_NM))/tau_NM
#print omg

mu_d_origin = 0.185 #0.185 #% 2.4; %0.2;
mu_d = mu_d_origin
grans_density = 6.3 * 10**8
Mss = monkey_weight * grans_density
#print Mss

Mss_EGFP = Mss * EGFP_ratio # * 0.4
#print EGFP_ratio, Mss, Mss_EGFP

num_appear_EGFP_clones = 2900

S_EGFPs = [60900,57750,87000,36750,71400,42750,51300,48000,139500]
S_EGFP = np.mean(S_EGFPs[1:])

r_h1 = 8.
############### load parameters ####################
shrink_ratio = 1.
tot_time = (months[-1] * 30 + 1) * shrink_ratio
dt = 0.1

thres_num_detect = Mss_EGFP/S_EGFP # TODO

thres_large = 0.0213 # TODO
thres_small = 50./S_EGFP # TODO

def my_euclid(v1, v2, sds = None):
    CONST_PENALTY = 10.

    v1 = np.array(v1)
    v2 = np.array(v2)

    masks_val = (v1 == v1) & (v2 == v2)
    v1_val = v1[masks_val]
    v2_val = v2[masks_val]

    penalty = (len(v1) - sum(masks_val)) * CONST_PENALTY

    if sds == None:
        return euclidean(v1_val, v2_val) + penalty
    else:
        sds_val = np.array(sds)[masks_val]
        v1_val = [v1_val[i]/sds_val[i] for i in range(len(v1_val))]
        v2_val = [v2_val[i]/sds_val[i] for i in range(len(v2_val))]
        #print euclidean(v1_val, v2_val)
        return euclidean(v1_val, v2_val) + penalty

def diff_cum(cum1, cum2):
    max_diff = 0.
    for i in range(len(cum1)):
        max_diff = max(abs(cum1[i]-cum2[i])*exp(i*0.01), max_diff)
    return max_diff

def get_expr_data(CELL_TYPE = 'Grans', SKIP_MONTH2 = False):
    input_all = pd.read_csv('data_Gr_cleaned.txt', sep='\t')
    #colnames = input_all.columns.tolist()

    #columns_grans = []
    #for colname in colnames:
    #    if CELL_TYPE in colname:
    #        columns_grans.append(colname)

    rowNum_grans = input_all.shape[0] # default setting
    for i in range(rowNum_grans):
        if input_all.loc[i][0] != input_all.loc[i][0]:
            rowNum_grans = i
            break

    data_grans_origin = input_all.loc[0:rowNum_grans-1]
    if SKIP_MONTH2:
        data_grans = data_grans_origin.ix[:,2:] # abandon sample at month 1,2
    else:
        data_grans = data_grans_origin
    data_grans = data_grans.loc[data_grans.sum(axis=1)>0] # abandon samples of all 0s
    data_expr = data_grans.values # numpy array
    return data_expr

########## Relative aspect ##########
def look_relative_stats(data_mat, cs, datalabel, step=1, num_bins = 50): #
    ################
    # Input: sample matrix, color and symbol, datalabel
    # Output: plotted error bars (unshown)
    ################
    data_X = data_mat[:,:-step] # f_i,j
    data_X_vec = data_X.reshape(data_X.size)

    bnds = np.array([exp(x) for x in np.linspace(log(0.00001),log(0.14),num_bins+1)])

    bins_inds = np.digitize(data_X_vec, bnds)
    X_bins = (bnds[:-1]+bnds[1:])/2

    data_Y = data_mat[:,step:]
    data_Y_vec = data_Y.reshape(data_Y.size)

    Y_binned_avg = [0]*num_bins
    Y_binned_std = [0]*num_bins

    for i in range(num_bins):
        curr_vec = data_Y_vec[bins_inds==i+1]
        Y_binned_avg[i] = np.mean(curr_vec)
        Y_binned_std[i] = np.std(curr_vec)

    plt.errorbar(X_bins, Y_binned_avg, yerr=Y_binned_std, color=cs, label=datalabel)
    return Y_binned_avg


########## Relative aspect ##########


########## Absolute aspect ##########
def get_fz_stats(data_mat):
    ################
    # Input: sample matrix
    # Output: for each z_i, get the avg & std abundances
    ################
    zi = np.array([0.] * len(data_mat))
    avgs = np.array([0.] * len(data_mat))
    for i in range(len(data_mat)):
        zi[i] = sum(data_mat[i,:]< 10.**-4) #TODO!
        avgs[i] = data_mat[i,:].mean()

    avgj_avg = np.array([0.] * (data_mat.shape[1]+1))
    avgj_std = np.array([0.] * (data_mat.shape[1]+1))
    for j in range(data_mat.shape[1]+1):
        avgj_avg[j] = avgs[zi==j].mean()
        avgj_std[j] = avgs[zi==j].std()

    return [avgj_avg, avgj_std]

def get_fsi_z(data_mat):
    ################
    # Input: sample matrix
    # Output: for each z_i, get the avg & std abundances
    ################
    zi = np.array([0.] * len(data_mat))
    avgs = np.array([0.] * len(data_mat))
    for i in range(len(data_mat)):
        zi[i] = sum(data_mat[i,:]==0)
        avgs[i] = data_mat[i,:].mean()

    fsi_z = []
    for j in range(data_mat.shape[1]+1):
        fsi_z.append(avgs[zi==j])
    fsi_z = np.array(fsi_z)
    return fsi_z

########## Absolute aspect ##########
def get_L(A_star):
    return log((Mss_EGFP/A_star)/(omg/(omg+mu_L)/mu_d))/log(2)

def get_bet(L):
    return (omg/(omg+mu_L)/mu_d) * 2**L

def get_A(L):
    return Mss_EGFP / (2**L * (omg/(omg+mu_L)/mu_d))

def get_pulse_alter(Ass, pulse_maxtime = 60):
    L = get_L(Ass)
    #L = log((Mss_EGFP/Ass)/(omg/(omg+mu_L)/mu_d))/log(2)

    ts_p = np.linspace(0,pulse_maxtime,int(round(pulse_maxtime/dt)))
    ns = [exp((r - mu_p - omg)*t)*(1-gammainc(L+1,2*r*t)) for t in ts_p]
    ms = [omg/(mu_d+r-mu_p-omg) * (n-exp(-mu_d*t) *
            (1-gammainc(L+1,(r+omg+mu_p-mu_d)*t)*2*r/(r+omg+mu_p-mu_d))) for t,n in zip(ts_p,ns)]
    plt.plot(ts_p,ms)
    plt.show()

def get_pulse(Ass, pulse_maxtime = 60, dt=dt):
    L = get_L(Ass) #log((Mss_EGFP/Ass)/(omg/(omg+mu_L)/mu_d))/log(2)
    #print L, Mss_EGFP/Ass

    # Prepare pulse function

    ts_p = np.linspace(0,pulse_maxtime,int(round(pulse_maxtime/dt)))

    nL_one = [(2*r/(r+mu_p-omg-mu_L)) ** L * gammainc(L, (r+mu_p-omg-mu_L)*t)\
                    * exp(-(omg+mu_L)*t) for t in ts_p]
    term_decay = [exp(-mu_d * t) for t in ts_p]
    ms_conv = np.convolve(nL_one, term_decay)*omg*dt
    ms_one_pulse = ms_conv[:len(ts_p)]

    return ms_one_pulse

def show_one_pulse(Ass, pulse_maxtime = 60, dt=dt):
    ms_one_pulse = get_pulse(Ass, pulse_maxtime = 60, dt=dt)
    freq_pulse_sampled = [binomial(S_EGFP, ms_t/Mss_EGFP)/S_EGFP for ms_t in ms_one_pulse]

    ts_p = np.linspace(0,pulse_maxtime,int(round(pulse_maxtime/dt)))

    fig, ax1 = plt.subplots()

    color_ms = 'r'
    ax1.plot(ts_p, ms_one_pulse, color=color_ms, linewidth=2) # plot ms_one_pulse
    for tl in ax1.get_yticklabels():
        tl.set_color(color_ms)
    #fig.rc('text', usetex=True)
    ax1.set_xlabel('day', fontsize=18)
    ax1.set_ylabel(r'$m_i$', fontsize=18)
    ax1.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax2 = ax1.twinx()
    color_prob = 'g'
    ax2.plot(ts_p, freq_pulse_sampled, color=color_prob) # plot freq_pulse_sampled
    for tl in ax2.get_yticklabels():
        tl.set_color(color_prob)
    ax2.set_xlabel('day', fontsize=18)
    ax2.set_ylabel('sampling frequency (one realization)', fontsize=18)
    ax2.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()
    #

############### Simulation parameters ####################
def simulate_HSC_dynamics(f_hi, Ass, Hss_EGFP = 2000., mu_h=0, tot_HSC_dynamics_time=tot_time, RETURN_HI=False):
    ## Input: f_hi, Hss_EGFP, alp
    ## Output: inj_times

    curr_time = 0
    inj_times = []
    hi_recs = []
    event_times = []
    hi = int(round(Hss_EGFP * f_hi))

    alp = Ass/Hss_EGFP

    while curr_time < tot_HSC_dynamics_time and hi > 0:
        event_times.append(curr_time)
        hi_recs.append(hi)

        rate_inj = alp * hi
        rate_birth = hi * mu_h
        rate_death = hi * mu_h
        rate_all = rate_inj + rate_birth + rate_death

        #print hi
        curr_time += log(1/(1-random.random()))/rate_all # random() can reach 0.0

        test_num = rate_all * random.random()
        if test_num <= rate_inj:
            inj_times.append(curr_time)
            continue
        test_num -= rate_inj

        if test_num <= rate_birth:
            hi += 1
            continue
        test_num -= rate_birth

        if test_num <= rate_death:
            hi -= 1
            continue
        test_num -= rate_death

        ## Not supposed to be here
        print 'Probablity overflow', rate_inj

    if RETURN_HI:
        return (inj_times, event_times, hi_recs)
    else:
        return inj_times

def get_ms(inj_times, ms_one,tot_time,dt):
    ts = np.linspace(0,tot_time,int(round(tot_time/dt)))
    ms = [0] * len(ts)

    for inj_time in inj_times:
        inj_ind = int(round(inj_time/dt))
        inj_len = min(len(ms_one), len(ms[inj_ind:]))
        for j in range(inj_len):
            ms[inj_ind+j] += ms_one[j]
    return ms

def get_samples(ms, sample_months):
    # Sampling
    sample_num = len(sample_months)
    samples_theo = [0] * sample_num #len(sample_months)
    for j in range(sample_num):
        sample_ind = int(round(sample_months[j]*shrink_ratio*30/dt))
        samples_theo[j] = binomial(S_EGFP, ms[sample_ind]/Mss_EGFP)/S_EGFP
    return samples_theo

## Input: f_hi and Ass
## Output: f_si
def simulate_freq(f_hi,
                  Ass,
                  Hss_EGFP = 2000.,
                  type=1,
                  mu_d = mu_d_origin,
                  sample_num=8,
                  use_expr_months = True,
                  pulse_maxtime = 60):

    if use_expr_months:
        sample_months = months #[1:sample_num+1]
        tot_sample_day = tot_time # * 30 #
    else:
        sample_months = [x*6 for x in range(sample_num)]
        tot_sample_day = sample_months[-1] * 30


    if f_hi == 0:
        return [0] * len(sample_months)

    L = log((Mss_EGFP/Ass)/(omg/(omg+mu_L)/mu_d))/log(2)

    ms_one = get_pulse(Ass=Ass, pulse_maxtime=pulse_maxtime, dt=dt)

    inj_times = simulate_HSC_dynamics(f_hi, Ass, Hss_EGFP, mu_h=0, tot_HSC_dynamics_time=tot_sample_day)
    #plt.plot(inj_times)
    #plt.show()

    ms = get_ms(inj_times, ms_one,tot_sample_day,dt)

    if type == 2:
        return np.array([x/Mss_EGFP for x in ms]).std()

    samples_theo = get_samples(ms,sample_months=sample_months)
    return samples_theo

def get_ck_cumu(samples, bins=None, GET_CUMU=True):
    samples = np.array(samples) # make sure it is a numpy object
    if samples.ndim == 1:
        row, col = len(samples), 1
    else:
        row, col = samples.shape

    if bins == None:
        bins = np.linspace(0,samples.max(),num=int(samples.max())+1)

    results = []
    row, col = samples.shape
    for j in range(col):

        curr_si = np.array([x*S_EGFP for x in samples[:,j]])
        binned_cnts = []
        for k in range(1,len(bins)):
            print bins[k-1], bins[k]
            binned_cnts.append(len(curr_si[(curr_si>=bins[k-1])&(curr_si<bins[k])]))
        if GET_CUMU:
            results.append(np.cumsum(binned_cnts))
        else:
            results.append(binned_cnts)
    return results

def get_ck_norm_cumu(samples, bnds=None, GET_CUMU=True):
    samples = np.array(samples) # make sure it is a numpy object
    row = len(samples)

    #if bins == None:
    #    bins = np.linspace(0,samples.max(),num=int(samples.max())+1)

    results = []
    #row, col = samples.shape

    curr_fsi = samples
    binned_cnts = []
    for k in range(1,len(bnds)):
        #print bins[k-1], bins[k]
        binned_cnts.append(len(curr_fsi[(curr_fsi>=bnds[k-1])&(curr_fsi<bnds[k])]))

    #print binned_cnts
    cum_cnts = np.cumsum(binned_cnts)
    if GET_CUMU:
        results = [float(x)/cum_cnts[-1] for x in cum_cnts]
    else:
        results = [float(x)/cum_cnts[-1] for x in binned_cnts]
    return results

def use_sci_nota(axis, usesci=True):
    ax = plt.gca()

    if axis == 'x':
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        if usesci:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        if usesci:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

[FONTSIZE_MATH,FONTSIZE_TEXT,FONTSIZE_TICK,FONTSIZE_LEGD] = [30,30,24,24]
[MARGIN_LEFT,MARGIN_RIGHT,MARGIN_BOTTOM,MARGIN_TOP] = [.18,.87,.15,.9]

def config_plot(xlab='', ylab='', legend_loc=(0.,0.), legend_prop=18):

    plt.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
    plt.rc('font', size=FONTSIZE_TICK-2, weight='bold')
    plt.tick_params(labelsize=FONTSIZE_TICK)

    x_is_math = '$' in xlab
    y_is_math = '$' in ylab

    #ax = plt.gca()
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))

    plt.legend(bbox_to_anchor=(legend_loc),bbox_transform=plt.gcf().transFigure,
               fontsize=FONTSIZE_LEGD, prop={'size':legend_prop})

    plt.xlabel(xlab, fontsize=FONTSIZE_MATH if x_is_math else FONTSIZE_TEXT)
    plt.ylabel(ylab, fontsize=FONTSIZE_MATH if y_is_math else FONTSIZE_TEXT)

    plt.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM,
                        right=MARGIN_RIGHT, top=MARGIN_TOP)
    #plt.tight_layout()

def calc_ck_cor():
    ''

def my_int(a_num):
    return int(round(a_num))

def gen_fhi(num_EGFP_clones, H, tag_ratio):
# Procedure:
# 1.construct c_k prob for whole Hss pool
# 2.randomly choose sizes for all clones
# 3.select 2%~3% of these clones as EGFP+ ones
    C = my_int(num_EGFP_clones/tag_ratio)

    lamb = 1 - float(C)/H
    print lamb

    #print 'lambda = ' + str(lamb)
    #bin_widths = [(1-lamb)**2*lamb**(k-1)*H/C for k in range(1,H+1)]
    bin_widths = [(1-lamb)*lamb**(k-1) for k in range(1,int(H)+1)] # distr for non-zero clones
    bnds = np.cumsum(bin_widths)

    rand_nums = [random.random() for _ in range(C)]

    rand_nums_EGFP = random.sample(rand_nums, num_EGFP_clones)
    bin_inds = np.digitize(rand_nums_EGFP, bnds) # 0 corresponds to size-1 clone, and so on
    #bin_inds = bin_inds[ bin_inds<len(bin_widths) ] # beyond this range
    hi_s = [x+1 for x in bin_inds]
    #print sum(hi_s), tag_ratio * H

    tot_hi_EGFP = sum(hi_s)
    fhi = [float(x)/tot_hi_EGFP for x in hi_s]

    return fhi


def get_appear_clone(samples):
    samples = np.array(samples)
    row, col = samples.shape
    num_appear_clones_months = []
    for j in range(col):
        num_appear_clones_months.append(sum(samples[:,j] > 0))

    num_appear_clones_total = 0
    for i in range(row):
        num_appear_clones_total += sum(samples[i,:]) > 0
    print num_appear_clones_total
    return num_appear_clones_months

#H0_EGFP_SANGGU = 11.*10**6 * 0.01/100
#H0_EGFP_SANGGU = my_int(H0_EGFP_SANGGU)
#print H0_EGFP_SANGGU

#num_EGFP_clones = 500
#H = 30000
#hi_s = gen_fhi(num_EGFP_clones,H,EGFP_ratio)
#print sum(hi_s), H * EGFP_ratio
#print len(hi_s)

#plt.plot(np.bincount(hi_s))
#plt.show()

def afterResults(A_star):
    beta = Mss_EGFP/A_star
    L = log(beta/(omg/(omg+mu_L)/mu_d))/log(2.)
    print 'beta: ', beta
    print 'L:', L

def show_pulses_ages(L, pulse_maxtime = 60, dt=dt):
    ts_p = np.linspace(0,pulse_maxtime*2,int(round(pulse_maxtime/dt)))
    nl = [[0.] * len(ts_p) for _ in range(L+1)]
    nl[0] = [exp(-(r+mu_p)*t) for t in ts_p]
    for l in range(1,L): # 0 <= l <= L
        nl[l] = [2*r*ts_p[j]/l * nl[l-1][j] for j in range(len(ts_p))]
    nl[L] = [(2*r/(r+mu_p-omg-mu_L)) ** L * gammainc(L, (r+mu_p-omg-mu_L)*t)\
                    * exp(-(omg+mu_L)*t) for t in ts_p]
    nl = np.array(nl)

    term_decay = [exp(-mu_d * t) for t in ts_p]
    ms_conv = np.convolve(nl[L], term_decay)*omg*dt
    ms_one_pulse = ms_conv[:len(ts_p)]

    fig = plt.figure(figsize=(8,6))
    #for i in range(L):
    #    plt.plot(ts_p, nl[i,:], linewidth=1.)
    #plt.plot(ts_p, nl[L,:], linewidth=1.5)
    plt.plot(ts_p[:len(ts_p)/2], ms_one_pulse[:len(ts_p)/2], 'b', linewidth=2)
    #plt.plot(ts_p[:len(ts_p)/2], ms_one_pulse[:len(ts_p)/2]*.5, 'g', linewidth=2)

    plt.plot(ts_p[len(ts_p)/2:], ms_one_pulse[:len(ts_p)/2], 'g', linewidth=2)
    #plt.plot(ts_p[len(ts_p)/2:], ms_one_pulse[:len(ts_p)/2]*.5, 'g', linewidth=2)
    config_plot('day', 'number of cells')
    use_sci_nota('x',usesci=False)
    use_sci_nota('y',usesci=True)
    plt.ylim(0,.5*10**6)
    plt.show()

    PLOT_INSET = False
    if PLOT_INSET:
        fig = plt.figure(figsize=(8,6))
        plt.plot(ts_p, ms_one_pulse, color='purple', linewidth=2.5)
        plt.axis('off')
        plt.show()