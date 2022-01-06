import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from scipy.special import erf
from helper import cache_grendel, cropsave, grendel_dir



"""
Comparison of power spectra between
CONCEPT and GADGET in 4 boxes:
2048, 1024, 512, 256
All using 1024³ particles.
"""



textwidth = 504  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 4.185
# The general font size is 9 but in captions it is 8.
# We choose to match this exactly.
fontsize = 8  #9/1.2

latex_preamble = r'''
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{mathtools}
    \usepackage{siunitx}
    \usepackage{xfrac}
    \usepackage{nicefrac}
    \usepackage{relsize}
    \newcommand{\CONCEPT}{\textsc{co\textsl{n}cept}}
    \newcommand{\CONCEPTONE}{\textsc{co\textsl{n}cept}\,\textscale{.77}{{1.0}}}
    \newcommand{\GADGET}{\textsc{gadget}}
    \newcommand{\GADGETTWO}{\textsc{gadget}-\textscale{.77}{{2}}}
    \newcommand{\GADGETFOUR}{\textsc{gadget}-\textscale{.77}{{4}}}
'''
matplotlib.rcParams.update({
    'text.usetex'        : True,
    'font.family'        : 'serif',
    'font.serif'         : 'cmr10',
    'font.size'          : fontsize,
    'mathtext.fontset'   : 'cm',
    'axes.formatter.use_mathtext': True,
    'text.latex.preamble': latex_preamble,
})



h = 0.67
size = 1024
N = size**3
z_values = [10, 5, 3, 2, 1, 0.5, 0]
a_values = [1/(1 + z) for z in z_values]
boxsizes = [2048, 1024, 512, 256]
nprocs = {2048: 256, 1024: 256, 512: 256, 256: 512}
concept_standard = ['', 'final', 'symplectic_final'][2]
gadget_standard = ['', ][0]
highprec_special = [  # for gadget
    'highprec',
    'ErrTolForceAcc0.001_TreeDomainUpdateFrequency0.05',
][1]
concept_special = [  # not used
    'dloga_dlogt',
][0]
lw = [
    None,  # default, corresponds to 1.5
    1.3,
][0]

# Load
cache_assume_uptodate = True
output_dir = f'{grendel_dir}/powerspec/concept_vs_gadget'
def load_concept(boxsize, special=''):
    directory = f'{output_dir}/nprocs{nprocs[boxsize]}/{boxsize}/{special}'.rstrip('/')
    P = []
    k = None
    for a in a_values:
        filename = f'{directory}/powerspec_a={a:.2f}'
        if cache_assume_uptodate:
            filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
        else:
            if not os.path.isfile(filename):
                continue
            filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
        k, _P = np.loadtxt(filename, usecols=(0, 2), unpack=True)
        P.append(_P)
    return k, P
def load_gadget_power(box, special=''):
    directory = f'{output_dir}/Gadget2/box{box}_size{size}/{special}'.rstrip('/')
    P = []
    k = None
    for i, a in enumerate(a_values):
        filename = f'{directory}/powerspec_snapshot_00{i}'
        if cache_assume_uptodate:
            filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
        else:
            if not os.path.isfile(filename):
                continue
            filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
        k, _P = np.loadtxt(filename, usecols=(0, 2), unpack=True)
        P.append(_P)
    return k, P

k_all = {}
P_concept = {}
P_gadget = {}
for boxsize in boxsizes:
    k_all[boxsize], P_concept[boxsize] = load_concept(boxsize, concept_standard)
    _, P_gadget[boxsize] = load_gadget_power(boxsize, gadget_standard)

P_gadget_highprec = {}
for boxsize in boxsizes:
    _, P_gadget_highprec[boxsize] = load_gadget_power(boxsize, highprec_special)

def k_nyq_particles(boxsize):
    """Supply box in [Mpc/h] to get k_Nyquist in [1/Mpc]
    """
    return 2*np.pi/boxsize*(np.cbrt(N)/2)*h

def smooth(x, y, n=500, num=40):
    fac = np.log(1024)/np.log(2048)
    num *= fac
    num = int(round(num))
    if num%2 == 0:
        num += 1
    x_interp = np.logspace(np.log10(x[0]), np.log10(x[-1]), n)
    y_interp = np.interp(np.log10(x_interp), np.log10(x), y)
    y_smoothed = scipy.signal.savgol_filter(y_interp, num, 2)
    #steepness = 2
    #k_smooth = 0.4*np.sqrt(x[0]*x[-1])
    #weight = (1 - erf(steepness*np.log10(x_interp/k_smooth)))/2
    #y_smoothed = weight*y_interp + (1 - weight)*y_smoothed
    return x_interp, y_smoothed

# Plot
fig, axes = plt.subplots(2, 2, figsize=(width, height))
textlabel_y = 0.770 #0.834
textlabels = {
    2048: (0.94, textlabel_y),
    1024: (0.94, textlabel_y),
     512: (0.795, textlabel_y),
     256: (0.70, textlabel_y),
}
do_smoothing = True
boxsize_legends = {2048: [1, 0.5, 0, 2], 1024: [10, 5, 3], 512: [], 256: []}
legend_locations = {2048: 'upper left', 1024: 'lower left', 512: 'upper left', 256: 'upper left'}
def get_mask(k, k_nyq):
    mask = (k < k_nyq)
    for i, el in enumerate(reversed(mask)):
        if el:
            mask[i-3:] = False
            break
    return mask
for boxsize, ax in zip(boxsizes, axes.flat):
    k = k_all[boxsize]
    if k is None:
        continue
    k_nyq = k_nyq_particles(boxsize)
    mask = get_mask(k, k_nyq)
    miny_all = +np.inf
    maxy_all = -np.inf
    any_labels = False
    for j, P_g in enumerate((P_gadget, P_gadget_highprec)):
        linetype = {0: '-', 1: '--'}[j]
        if not P_g:
            continue
        n = np.min((len(P_concept[boxsize]), len(P_g[boxsize])))
        n_special = n  # np.min((len(P_concept_special[boxsize]), len(P_g[boxsize])))
        for i in range(np.max((n, n_special))):
            z = z_values[i]
            if j == 1 and z > 1 and z != 10:  # Only plot late highprec gadget
               continue
            rel = P_concept[boxsize][i]/P_g[boxsize][i]
            label = None
            #if j == 0 and z in boxsize_legends[boxsize]:
            #    any_labels = True
            #    z_str = str(z)
            #    if z == 0.5:
            #        z_str = r'\text{\textonehalf}'
            #    label = f'$z = {z_str}$'
            y = (rel[mask] - 1)*100
            miny = np.min(y)
            if miny < miny_all:
                miny_all = miny
            maxy = np.max(y)
            if maxy > maxy_all:
                maxy_all = maxy
            x = k[mask]/h
            if do_smoothing:
                x, y = smooth(k[1:]/h, (rel[1:] - 1)*100)
            ax.semilogx(x, y, f'C{i}{linetype}', label=label, lw=lw, zorder=(-30 + i - 0.1*j))
    ax.fill_between(k/h, np.ones(k.size), -np.ones(k.size),
        color=[0]*3, ec=None, zorder=-100, alpha=0.052)
    ax.fill_between(k/h, 0.5*np.ones(k.size), -0.5*np.ones(k.size),
        color=[0]*3, ec=None, zorder=-100, alpha=0.0565)
    ax.fill_between(k/h, 0.1*np.ones(k.size), -0.1*np.ones(k.size),
        color=[0]*3, ec=None, zorder=-100, alpha=0.065)
    ax.semilogx(k/h, np.zeros(k.size), 'k-', lw=0.5, zorder=-99)
    # Note: The global offset is about 0.1%,
    # but gets smaller with decreasing boxsize.
    #ax.semilogx(k/h, 0.1*np.ones(k.size), 'k-', lw=0.5, zorder=99)
    ax.set_xlim(k[1]/h, k_nyq/h)
    lim = 1.0
    if miny_all > -lim:
        miny_all = -lim
    elif miny_all < -1.5:
        miny_all = -1.5 + 1e-9
    if maxy_all < lim:
        maxy_all = lim
    elif maxy_all > 1.5:
        maxy_all = 1.5 - 1e-9
    ax.set_ylim(miny_all - 0.05*(maxy_all - miny_all), maxy_all + 0.05*(maxy_all - miny_all))
    #ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.text(*textlabels[boxsize], rf'$L_{{\text{{box}}}} = \SI{{{boxsize}}}{{Mpc}}/h$',
        va='center', ha='right', transform=ax.transAxes)
    #if any_labels:
    #    ax.legend(loc=legend_locations[boxsize], framealpha=0.6)
# Reverse legend order
#handles, labels = axes[0,0].get_legend_handles_labels()
#axes[0,0].legend(handles[::-1], labels[::-1], loc='lower left', framealpha=0.6)
#handles, labels = axes[1,0].get_legend_handles_labels()
#axes[1,0].legend(handles[::-1], labels[::-1], loc='upper left', framealpha=0.6)

legend1 = axes[0, 0].legend(
    (
          axes[0, 0].plot(0.1, 0, 'C6-', lw=lw)
        + axes[0, 0].plot(0.1, 0, 'C5-', lw=lw)
        + axes[0, 0].plot(0.1, 0, 'C4-', lw=lw)
        + axes[0, 0].plot(0.1, 0, 'C3-', lw=lw)
#        + axes[0, 0].plot(0.1, 0, 'C2-', lw=lw)
#        + axes[0, 0].plot(0.1, 0, 'C1-', lw=lw)
#        + axes[0, 0].plot(0.1, 0, 'C0-', lw=lw)
    ),
    (
        r'$z = 0$',
        r'$z = \text{\textonehalf}$',
        r'$z = 1$',
        r'$z = 2$',
        #r'$z = 3$',
        #r'$z = 5$',
        #r'$z = 10$',
    ),
    loc='upper left',
    framealpha=0.6,
)
legend2 = axes[0, 0].legend(
    (
        #  axes[0, 0].plot(0.1, 0, 'C6-', lw=lw)
        #+ axes[0, 0].plot(0.1, 0, 'C5-', lw=lw)
        #+ axes[0, 0].plot(0.1, 0, 'C4-', lw=lw)
        #+ axes[0, 0].plot(0.1, 0, 'C3-', lw=lw)
          axes[0, 0].plot(0.1, 0, 'C2-', lw=lw)
        + axes[0, 0].plot(0.1, 0, 'C1-', lw=lw)
        + axes[0, 0].plot(0.1, 0, 'C0-', lw=lw)
    ),
    (
        #r'$z = 0$',
        #r'$z = \text{\textonehalf}$',
        #r'$z = 1$',
        #r'$z = 2$',
        r'$z = 3$',
        r'$z = 5$',
        r'$z = 10$',
    ),
    loc='lower left',
    framealpha=0.6,
)
axes[0, 0].add_artist(legend1)
axes[0, 0].add_artist(legend2)

dashdotted = (0, (4, 1, 1, 1))
#legend1 = axes[1, 0].legend(
#    (
#          axes[1, 0].plot(0.1, 0, '-', color='grey', lw=lw)
#    ),
#    (
#        r'\GADGETTWO{}',
#    ),
#    loc=(0.021, 0.54),  # 0.60
#    framealpha=0.6,
#)
legend2 = axes[1, 0].legend(
    (
          axes[1, 0].plot(0.1, 0, '-' , color='grey', lw=lw)
        + axes[1, 0].plot(0.1, 0, '--', color='grey', lw=lw)
        + axes[1, 0].plot(0.1, 0, ':' , color='grey', lw=lw)
        #+ axes[1, 0].plot(0.1, 0, linestyle=dashdotted, color='grey', lw=lw)
    ),
    (
        r"`standard' precision",        
        r'high precision',
        r'high precision, $x_{\text{r}} = 5.5x_{\text{s}}$',
    ),
    loc='lower left',
    framealpha=0.6,
)
#axes[1, 0].add_artist(legend1)
axes[1, 0].add_artist(legend2)

axes[0,0].xaxis.tick_top()
axes[0,0].xaxis.set_label_position('top')
axes[0,1].xaxis.tick_top()
axes[0,1].xaxis.set_label_position('top')
axes[0,1].yaxis.tick_right()
axes[1,1].yaxis.tick_right()
axes[0,1].yaxis.set_label_position('right')
axes[1,1].yaxis.set_label_position('right')
for ax in axes.flat:
    ax.set_xlabel(r'$k\; [h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\displaystyle \frac{P_{\text{\CONCEPTONE{}}}}{P_{\text{\GADGETTWO{}}}} - 1\; [\%]$')
    #ax.set_ylabel(r'$P_{\text{\CONCEPTONE{}}}/P_{\text{\GADGETTWO{}}} - 1\; [\%]$')
axes[0,0].set_xticks([0.01, 0.1, 1])
axes[0,0].set_xticklabels([r'$0.01$', r'$0.1$', r'$1$'])
axes[0,1].set_xticks([0.01, 0.1, 1])
axes[0,1].set_xticklabels([r'$0.01$', r'$0.1$', r'$1$'])
axes[1,0].set_xticks([0.1, 1])
axes[1,0].set_xticklabels([r'$0.1$', r'$1$'])
axes[1,1].set_xticks([0.1, 1, 10])
axes[1,1].set_xticklabels([r'$0.1$', r'$1$', r'$10$'])
for ax in axes.flat:
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels([r'$-1$', r'$-1/2$', r'$0$', r'$1/2$', r'$1$'])

# xr = 5.5
for i_55 in [0, 1, 2, 6]: #, 6]:
    z_55 = z_values[i_55]
    a_55 = a_values[i_55]
    filename = f'{output_dir}/nprocs256/512/xs=1.25_xr=5.5/powerspec_a={a_55:.2f}'
    filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
    k, P_55_concept = np.loadtxt(filename, usecols=(0, 2), unpack=True)
    filename = f'{output_dir}/Gadget2/box512_size1024/xr5.5/powerspec_snapshot_00{i_55}'
    filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
    k, P_55_gadget = np.loadtxt(filename, usecols=(0, 2), unpack=True)
    filename = f'{output_dir}/Gadget2/box512_size1024/xr5.5_ErrTolForceAcc0.001_TreeDomainUpdateFrequency0.05/powerspec_snapshot_00{i_55}'
    filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
    k, P_55_gadget_highprec = np.loadtxt(filename, usecols=(0, 2), unpack=True)

    #axes[1, 0].semilogx(k_all[512]/h, (P_55_concept/P_gadget[512][i_55] - 1)*100, 'k-', lw=0.9)
    #axes[1, 0].semilogx(k_all[512]/h, (P_55_concept/P_55_gadget - 1)*100, 'y-', lw=0.9)
    #axes[1, 0].semilogx(k_all[512]/h, (P_55_concept/P_55_gadget - 1)*100, 'k:', zorder=+80)
    rel = P_55_concept/P_55_gadget_highprec 
    x, y = smooth(k_all[512][1:]/h, (rel[1:] - 1)*100)
    axes[1, 0].semilogx(x, y, f'C{i_55}:', lw=lw, zorder=-80)
    #axes[1, 0].semilogx(k_all[512]/h, (P_55_concept/P_gadget[512][i_55] - 1)*100, 'm-', lw=0.9)

# GADGET4
gadget_output = [
    #'512_ffm5pm_random_oldsrc',
    #'512_ffm5pm_random',
    #'512_ffm5pm_random_oldsrc_nonsyncdump',

    '512_tree2pm_nonsyncdump',
    '512_tree5pm_nonsyncdum',
    '512_ffm5pm_nonsyncdump',
    '512_ffm5pm_random_nonsyncdump',
    '512_ffm5pm_random_xr55_nonsyncdump',
][3]
for i_gadget4 in []: #[0, 1, 2, 3, 4, 5, 6]:
    filename = f'{output_dir}/Gadget4/{gadget_output}/snapdir_00{i_gadget4}/powerspec_snapshot_00{i_gadget4}'
    filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
    k, P_gadget4 = np.loadtxt(filename, usecols=(0, 2), unpack=True)
    rel = P_concept[512][i_gadget4]/P_gadget4
    x, y = smooth(k_all[512][1:]/h, (rel[1:] - 1)*100)
    axes[1, 0].semilogx(x, y, color=f'C{i_gadget4}', linestyle=dashdotted, lw=lw, zorder=1000)

# Hide ugly label bits
grey_bg_color = [0.99215686]*3
axes[1, 0].plot([0.116]*2, [0.08, 0.187], '-', color=grey_bg_color, transform=axes[1, 0].transAxes, zorder=np.inf, alpha=1)

# Reassign xticks to 512 box
axes[1, 0].set_xticks([0.1, 1])
axes[1, 0].set_xticklabels([r'$0.1$', r'$1$'])

# Same ylim for all axes?
for ax in axes.flat:
    ax.set_ylim(-0.92, 1.25)
    #ax.set_ylim(-0.92-1, 1.1+2)

#for boxsize, ax in zip(boxsizes, axes.flat):
#    k_div = 2*np.pi/boxsize*24.0
#    ax.plot([k_div/h]*2, [-2, 2], 'k')
#    print(k_div/h)

# Save
fig.subplots_adjust(wspace=0, hspace=0)
cropsave(fig, '../figure/correctness.pdf')  # no tight_layout() or bbox_inches()

