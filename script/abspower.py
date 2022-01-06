import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from classy import Class
from helper import cache_grendel, cropsave, grendel_dir



"""
For just one of the boxes, plot the absolute CONCEPT
and GADGET spectra over time, including a = a_begin.
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
    \newcommand{\GADGETTWO}{\textsc{gadget}-\textscale{.77}{{2}}}
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
boxsizes = [512]
boxsize = boxsizes[0]

nprocs = {2048: 256, 1024: 256, 512: 256, 256: 1024}
concept_standard = ['', 'final', 'symplectic_final'][2]
gadget_standard = ['', ][0]

textwidth = 240  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 2.09
fig = plt.figure(figsize=(width, height))
n_axis = 6
gs = fig.add_gridspec(n_axis, 1)
ax1 = fig.add_subplot(gs[:(n_axis - 1), 0])
ax2 = fig.add_subplot(gs[n_axis - 1, 0])
axes = [ax1, ax2]

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


def get_mask(k, k_nyq):
    mask = (k < k_nyq)
    for i, el in enumerate(reversed(mask)):
        if el:
            mask[i-3:] = False
            break
    return mask

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
    #weight = (1 - scipy.special.erf(steepness*np.log10(x_interp/k_smooth)))/2
    #y_smoothed = weight*y_interp + (1 - weight)*y_smoothed
    return x_interp, y_smoothed

# Also load initial power spectrum
filename = f'{grendel_dir}/powerspec/box512_size1024/powerspec_a=0.01'
filename = cache_grendel(filename, cache_assume_uptodate=cache_assume_uptodate)
k_ini, P_ini = np.loadtxt(filename, usecols=(0, 2), unpack=True)
k = k_all[boxsize]
if not np.all(k_ini == k):
    print('Mismatch between initial and sim k!', file=sys.stderr)
a_begin = 0.01
z_values_all = [int(round(1/a_begin - 1))] + z_values
def get_class(boxsize):
    k = k_all[boxsize]
    cosmo = Class()
    Omega_b = 0.049
    Omega_cdm = 0.27
    params = {
        'Omega_b': Omega_b,
        'Omega_cdm': Omega_cdm,
        'H0': 67.0,
        'P_k_max_1/Mpc': np.max(k)*1.01,
        'output': 'dTk mPk',
        'z_pk': ', '.join([str(float(z)) for z in z_values_all]),
    }
    cosmo.set(params)
    cosmo.compute()
    P_class = [np.array([cosmo.pk(ki, z) for ki in k]) for z in z_values_all]
    # Scale according to D(a) with and without radiation
    bg = cosmo.get_background()
    a_bg = 1/(1 + bg['z'])
    a_min = 1e-6
    mask = (a_bg >= a_min)
    a_bg = a_bg[mask]
    D_class = bg['gr.fac. D'][mask]
    Omega_m = Omega_b + Omega_cdm
    Omega_Lambda = 1 - Omega_m
    D_concept = a_bg*scipy.special.hyp2f1(1/3, 1, 11/6, -Omega_Lambda/Omega_m*a_bg**3)
    D_concept /= D_concept[-1]
    D_class_begin = scipy.interpolate.interp1d(np.log(a_bg), np.log(D_class), kind='cubic')(np.log(a_begin))
    D_concept_begin = scipy.interpolate.interp1d(np.log(a_bg), np.log(D_concept), kind='cubic')(np.log(a_begin))
    D_class = D_class * (D_concept_begin/D_class_begin)  # same D at a_begin
    facs = scipy.interpolate.interp1d(np.log(a_bg), D_concept/D_class, kind='cubic')(np.log([1/(1 + z) for z in z_values_all]))**2
    P_class = [P_cl*fac for P_cl, fac in zip(P_class, facs)]
    # Match at a_begin (difference in gauge)
    fac = P_ini/P_class[0]
    P_class = [P_cl*fac for P_cl in P_class]
    return P_class
k_nyq = k_nyq_particles(boxsize)
mask = get_mask(k, k_nyq)
P_class = get_class(boxsize)
def plot(i, ax, P_c, P_g, P_cl, clip_on=True):
    zorder = -100*(i+1)
    z = z_values_all[i]
    if z == 0.5:
        z = r'\text{\textonehalf}'
    color = f'C{i-1}'
    if i == 0:
        color = f'C{len(z_values_all)-1}'
    x, y = k[mask]/h, ((k/h)**1.5*P_c *h**3)[mask]
    x, y = smooth(x, np.log(y))
    y = np.exp(y)
    ax.loglog(x, y, f'{color}-',
        label=f'$z = {z}$', clip_on=clip_on, zorder=zorder)
    x, y = k[mask]/h, ((k/h)**1.5*P_g *h**3)[mask]
    x, y = smooth(x, np.log(y))
    y = np.exp(y)
    ax.loglog(x, y, f'k--', clip_on=clip_on, zorder=zorder)
    ax.loglog(k[mask]/h, ((k/h)**1.5*P_cl*h**3)[mask], f'k:', lw=1, clip_on=clip_on, zorder=zorder)
# a == a_begin
plot(0, ax2, P_ini, P_ini, P_class[0], clip_on=False)

# a > a_begin
for i, (P_c, P_g, P_cl) in enumerate(zip(P_concept[boxsize], P_gadget[boxsize], P_class[1:]), 1):
    plot(i, ax1, P_c, P_g, P_cl, clip_on=(i != 1))
    # Legend needs to be made from ax2
    z = z_values_all[i]
    if z == 0.5:
        z = r'\text{\textonehalf}'
    color = f'C{i-1}'
    if i == 0:
        color = f'C{len(z_values_all)-1}'
    ax2.loglog(k[mask][0], P_c[mask][0], f'{color}-', label=f'$z = {z}$')
legend1 = ax2.legend(framealpha=0.6)
handles, labels = ax2.get_legend_handles_labels()
jumpy = 0.24
jumpx = 0.0088
legend1 = ax2.legend(handles[::-1], labels[::-1], loc=(0.037 + jumpx, 0.228 + jumpy))
legend1.set_zorder(np.inf)
legend1.set_clip_on(False)

for ax in axes:
    ax.set_xlim(k[0]/h, k_nyq/h)
ax2.set_xlabel(r'$k\; [h/\mathrm{Mpc}]$')
ax1.set_ylabel(r'$k^{3/2}P\; [(\mathrm{Mpc}/h)^{3/2}]$ .......')
ax1.fill([-0.155, -0.115, -0.115, -0.155, -0.155], [0.78, 0.78, 0.98, 0.98, 0.78], 'w', ec='none', clip_on=False, transform=ax1.transAxes, zorder=np.inf)
fig.subplots_adjust(wspace=0, hspace=0.205)

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(which='both', top=False, labeltop=False)
ax2.xaxis.tick_bottom()

# ylim
ax1.set_ylim(0.185, 6.2e+2)
ax2.set_ylim(2e-3, 0.803e-2)

# Place cut-out slanted lines
mew = 0.8
q = 22*np.pi/180
offset = 0.33
x = np.linspace(offset*np.pi, (2 - offset)*np.pi, 100)
y = np.sin(x)
X, Y = np.array([[np.cos(q), -np.sin(q)], [np.sin(q), np.cos(q)]]) @ np.array([x, y])
X -= np.mean(X)
Y -= np.mean(Y)
scale = 0.01
ex = 0.0024
for xx in (0, 1):
    ax1.plot(xx + scale*X, (0 - ex) + scale*Y, 'k-',
        lw=mew, transform=ax1.transAxes, zorder=1e+6, clip_on=False)
    ax2.plot(xx + scale*X, (1 + ex) + scale*(n_axis - 1)*Y, 'k-',
        lw=mew, transform=ax2.transAxes, zorder=1e+6, clip_on=False)
    # Clean up
    if xx == 1:
        ax1.plot(xx + scale*X + 0.001, (0 - ex) + scale*Y - 0.008, 'w-', alpha=1,
            lw=mew, transform=ax1.transAxes, zorder=1e+6, clip_on=False)
# old
#d = 0.65  # proportion of vertical to horizontal extent of the slanted line
#kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8.5,
#              linestyle='none', color='k', mec='k', mew=mew, clip_on=False)
#ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
#ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# Clean a = a_begin curve at the lower right
ax2.plot([0.996, 1.004, 1.004], [-0.035, -0.035, 0.06], 'w-', lw=1.2, transform=ax2.transAxes, clip_on=False, zorder=-5, alpha=1)
# Clean a = a_begin curve at the lower left
ax2.plot([-0.004]*2, [0.6, 0.8], 'w-', lw=1.2, transform=ax2.transAxes, clip_on=False, zorder=-5, alpha=1)
# Clean a = 0.09 curve at the lower left
ax1.plot([-0.004]*2, [0.08, 0.15], 'w-', lw=1.2, transform=ax1.transAxes, clip_on=False, zorder=-5, alpha=1)

# Draw extra part of spine with tick at lower left
ax2.tick_params(which='both', labelleft=False)
ax2.plot(
    [ax2.get_xlim()[0]]*2,
    [ax2.get_ylim()[0], 1e-3],
    'k-', lw=mew, clip_on=False,
)
ax2.plot(
    [ax2.get_xlim()[0], 0.907*ax2.get_xlim()[0]], 
    [1e-3]*2,
    'k-', lw=mew, clip_on=False,
)
ax2.text(0.804*ax2.get_xlim()[0], 0.9356*1e-3, r'$10^{-3}$', ha='right', va='center')

# Extra legend
x_leg_start = 0.3445 + jumpx
legend2 = ax2.legend(
    (
          ax2.plot(1, 1, '-' , color='w')
        + ax2.plot(1, 1, '--' , color='k')
        + ax2.plot(1, 1, ':', lw=1, color='k')
    ),
    (r'\CONCEPTONE{}', r'\GADGETTWO{}', r'linear'),
    loc=(x_leg_start, 0.228 + jumpy), framealpha=0.6,

)
ax2.add_artist(legend1)
ax2.add_artist(legend2)
# Rainbow
y_bot = 1.8935
y_top = y_bot + 0.0862
offsetx = 0.0123
dx = 0.01102*8/len(z_values_all)
for i in range(len(z_values_all)):
    color = f'C{i-1}'
    if i == 0:
        color = f'C{len(z_values_all)-1}'
    ax2.fill(
        [
            x_leg_start + offsetx + dx*i*0.995,
            x_leg_start + offsetx + dx*(i+1)/0.995,
            x_leg_start + offsetx + dx*(i+1)/0.995,
            x_leg_start + offsetx + dx*i*0.995,
            x_leg_start + offsetx + dx*i*0.995,
        ],
        np.array([y_bot, y_bot, y_top, y_top, y_bot]) + jumpy,
        color, alpha=1.0,
        ec='none', transform=ax2.transAxes, zorder=np.inf, clip_on=False,
)
# Remove small legend bits
ax2.plot([0.443 + jumpx]*2, np.array([0.5, 1.5]) + jumpy, 'w', transform=ax2.transAxes, zorder=np.inf, clip_on=False, alpha=1)

ax2.set_xticks([0.1, 1])
ax2.set_xticklabels([r'$0.1$', r'$1$'])

# Save
cropsave(fig, '../figure/abspower.pdf')  # no tight_layout() or bbox_inches()

