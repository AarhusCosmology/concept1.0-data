import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import scipy.optimize
from helper import load, load_gadget, mean8, cropsave, get_factor_after_symplectifying, grendel_dir



"""
BOXSIZE SCALING

N = 512³, nprocs = 64.
Total computation time as function of box size,
shown for both CONCEPT and GADGET-2.
"""



textwidth = 240  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 2.64
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



# Load
cache_assume_uptodate = True
output_dir = f'{grendel_dir}/powerspec/boxsize_scaling/64'
gadget_logs_dir = f'{grendel_dir}/log/gadget2/boxsize_scaling/64'
infos_boxsize_time = load(output_dir, 'time', check_spectra=0, cache_assume_uptodate=cache_assume_uptodate)
infos_boxsize_gadget = load_gadget(output_dir, gadget_logs_dir, cache_assume_uptodate=cache_assume_uptodate)
sympletic_performancehit = get_factor_after_symplectifying()

# Plot
fig = plt.figure(figsize=(width, height))
boxsizes = sorted(list(infos_boxsize_time.keys()))
y = [0]*len(boxsizes)
for boxsize, info in infos_boxsize_time.items():
    y[boxsizes.index(boxsize)] = sympletic_performancehit*info['t_total']
boxsizes = np.array(boxsizes)
y = np.array(y)
#C0_dark = np.asarray(matplotlib.colors.ColorConverter().to_rgb('C0'), dtype=float)*0.7
#C1_dark = np.asarray(matplotlib.colors.ColorConverter().to_rgb('C1'), dtype=float)*0.75
plt.loglog(boxsizes, y/60**2, f'.', color='C1', zorder=101)
# Fit (exponential)
a, b = np.polyfit(1/boxsizes[3:], np.log(y[3:]), 1)
X = np.logspace(np.log10(1/4096), np.log10(1/128), 1000)
XX = np.logspace(np.log10(1/4096), np.log10(1/8), 1000)
y_fit = np.exp(b)*np.exp(a*XX)
#plt.loglog(1/XX, y_fit/60**2, 'r', zorder=0)
# Removal of load imbalance
y_perfect = []
for boxsize in boxsizes:
    info = infos_boxsize_time[boxsize]
    t_total = sympletic_performancehit*info['t_total']
    for ts in info['data']:
        t_shortrange_actual  = sympletic_performancehit*ts.t_shortrange
        t_shortrange_perfect = sympletic_performancehit*ts.t_shortrange/(np.max(ts.load_imbalance) + 1)
        t_total -= t_shortrange_actual
        t_total += t_shortrange_perfect
    y_perfect.append(t_total)
plt.loglog(boxsizes, np.array(y_perfect)/60**2, f'.', color='C1', zorder=101)
# Fit (exponential)
a, b = np.polyfit(1/boxsizes[3:], np.log(y_perfect[3:]), 1)
y_fit = np.exp(b)*np.exp(a*XX)
#plt.loglog(1/XX, y_fit/60**2, 'r', zorder=0)

# Sigmoid fit WITH LIN AT LOW END
xdata = 1/boxsizes
def f(logx, a1, a, b, c, x0):
    x = np.exp(logx)
    y = (a1*x + b) + (a - a1)/(1 + np.exp(-c*(x - x0)))*x
    return np.log(y)
dashes = ''
for ydata in (np.array(y), np.array(y_perfect)):
    dashes += '-'
    popt, pcov = scipy.optimize.curve_fit(f, np.log(xdata), np.log(ydata),
        [1, 1e+7, 1e+3, 6e+2, 0.006],
        bounds=(
            [0, 1e+5, 1e+2, 6e+1, -0.01],
            [1e+3, 1e+9, 1e+4, 6e+3, +0.01],
        ),
    )
    y_popt = np.exp(f(np.log(XX), *popt))
    label = None
    if dashes == '-':
        label = r'\CONCEPTONE{}'  # 'CO$N$CEPT'
    plt.plot(1/XX, y_popt/60**2, f'C1{dashes}', label=label, zorder=100)
    # Linear behaviour at ends
    a1, a, b, c, x0 = popt
    XXX = np.logspace(np.log10(1/128), np.log10(1/8), 200)
    #plt.plot(1/XXX, (a*XXX + b)/60**2, 'k-')

# Gadget
boxsizes_gadget = sorted(list(infos_boxsize_gadget.keys()))
y_gadget = [0]*len(boxsizes_gadget)
for boxsize, info in infos_boxsize_gadget.items():
    y_gadget[boxsizes_gadget.index(boxsize)] = info['t_total']
boxsizes_gadget = np.array(boxsizes_gadget)
y_gadget = np.array(y_gadget)
#plt.loglog(boxsizes_gadget, y_gadget/60**2, f'C0.', ms=1, label='GADGET-2', zorder=0)
plt.loglog(boxsizes_gadget, y_gadget/60**2, f'.', color='C0', zorder=1)
# Fit (sigmoid)
popt, pcov = scipy.optimize.curve_fit(f, np.log(xdata), np.log(y_gadget), [1, 1e+7, 1e+3, 8e+2, 0.004], bounds=([0, 1e+6, 1e+2, 8e+1, -0.008], [1e+3, 1e+8, 1e+4, 8e+3, +0.008]))
y_popt = np.exp(f(np.log(XX), *popt))
#plt.plot(1/XX, y_popt/60**2, 'C0-', label='GADGET-2', zorder=0)
# Fit (linear)
a_gadget, b_gadget = np.polyfit(1/boxsizes_gadget, y_gadget, 1)
y_fit_gadget = a_gadget*XX + b_gadget
#plt.loglog(1/XX, y_fit_gadget/60**2, 'C0-', zorder=0, label='GADGET-2')
def f_lin(logx, a, b):
    x = np.exp(logx)
    y = b + a*x
    return np.log(y)
popt, pcov = scipy.optimize.curve_fit(f_lin, np.log(xdata), np.log(y_gadget), [a_gadget, b_gadget], bounds=([a_gadget/10, b_gadget/10], [a_gadget*10, b_gadget*10]))
y_popt = np.exp(f_lin(np.log(XX), *popt))
plt.plot(1/XX, y_popt/60**2, 'C0-', zorder=0, label=r'\GADGETTWO{}') #'GADGET-2')

# Finalize
plt.legend(framealpha=0.6)
plt.xlabel('$L_{\mathrm{box}}\; [\mathrm{Mpc}/h]$')
plt.ylabel('total computation time [hr]')
ax_bot = plt.gca()
ax_top = plt.gca().twiny()

#ax_bot.set_xlim(boxsizes_gadget[0], boxsizes_gadget[-1])
boxsizes_axis = []
boxsize = 4096
while boxsize > 48:
    boxsizes_axis.append(boxsize)
    boxsizes_axis.append(boxsize//4*3)
    boxsize //= 2
boxsizes_axis.pop()
boxsizes_axis = np.array(boxsizes_axis[::-1])
ax_bot.set_xlim(boxsizes_axis[0], boxsizes_axis[-1])
ax_bot.set_xticks(boxsizes_axis)
ax_bot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_bot.tick_params(axis='x', which='minor', bottom=False)
ax_bot.invert_xaxis()
k_Nyquist = 2*np.pi/boxsizes_axis[::-1]*(512/2)
xticks = np.log(k_Nyquist)
xticks -= xticks[0]
xticks = [xtick/xticks[-1] for xtick in xticks]
xticklabels = [f'${k:.3g}$' for k in k_Nyquist]
ax_top.set_xticks(xticks)
ax_top.set_xticklabels(xticklabels, rotation=34, ha='left', rotation_mode='anchor')
ax_top.set_xlabel('$k_{\mathrm{Nyquist}}\; [h/\mathrm{Mpc}]$')
plt.ylim(9e+2/60**2, 1.5e+6/60**2)

ax_bot.set_xticklabels([f'${b}$' for b in boxsizes_axis],
    rotation=34, ha='right', rotation_mode='anchor')

# Save
cropsave(fig, '../figure/box.pdf')  # no tight_layout() or bbox_inches()

