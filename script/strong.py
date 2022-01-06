import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
from helper import load, mean8, cropsave, grendel_dir



"""
STRONG SCALING

This figure contains 3 panels:
- Upper left:  Computation time per step at z ~ 99.
- Upper right: Computation time per step at z ~  0.
- Lower: Computation time per step throughout simulation.
The total compuation time as function of nprocs should be
shown in a separate figure, together with Gadget times.
"""



textwidth = 504  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 4.54
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
box, size = 1024, 512
output_dir = f'{grendel_dir}/powerspec/nprocs_scaling/box{box}_size{size}'
infos_time = load(output_dir, 'time', cache_assume_uptodate=cache_assume_uptodate)
info_37 = infos_time.pop(37)
time_steps = np.arange(len(next(iter(infos_time.values()))['data']))

# Plot
fig = plt.figure(figsize=(width, height))
gs = matplotlib.gridspec.GridSpec(2, 2, figure=fig)
axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, :]),
]
# Upper panels
nprocs_all = list(infos_time.keys())
t_fft_early            = np.zeros(len(nprocs_all))
t_longrange_early      = np.zeros(len(nprocs_all))
t_shortrange_early     = np.zeros(len(nprocs_all))
t_fft_late            = np.zeros(len(nprocs_all))
t_longrange_late      = np.zeros(len(nprocs_all))
t_shortrange_late     = np.zeros(len(nprocs_all))
for c, (nprocs, info) in enumerate(infos_time.items()):
    color = f'C{c}'
    t_fft          = np.array([info['data'][i].t_fft          for i in time_steps])
    t_longrange    = np.array([info['data'][i].t_longrange    for i in time_steps])
    t_shortrange   = np.array([info['data'][i].t_shortrange   for i in time_steps])
    load_imbalance = np.array([info['data'][i].load_imbalance for i in time_steps])
    index = nprocs_all.index(nprocs)
    t_fft_early       [index] = mean8(t_fft       , n_steps=len(time_steps))[ 0 + 8]
    t_longrange_early [index] = mean8(t_longrange , n_steps=len(time_steps))[ 0 + 8]
    t_shortrange_early[index] = mean8(t_shortrange, n_steps=len(time_steps))[ 0 + 8]
    t_fft_late        [index] = mean8(t_fft       , n_steps=len(time_steps))[-1 - 8]
    t_longrange_late  [index] = mean8(t_longrange , n_steps=len(time_steps))[-1 - 8]
    t_shortrange_late [index] = mean8(t_shortrange, n_steps=len(time_steps))[-1 - 8]
axes[0].loglog(nprocs_all, t_longrange_early + t_shortrange_early,
    'C3.-', zorder=10)
axes[0].loglog(nprocs_all, t_shortrange_early,
    'C0.-', zorder=9)
axes[0].loglog(nprocs_all, t_longrange_early,
    'C1.-', zorder=8)
axes[0].loglog(nprocs_all, t_fft_early,
    'C1.--', zorder=7)
axes[1].loglog(nprocs_all, t_longrange_late + t_shortrange_late,
    'C3.-', zorder=10)
axes[1].loglog(nprocs_all, t_shortrange_late, 'C0.-', zorder=9)
axes[1].loglog(nprocs_all, t_longrange_late, 'C1.-', zorder=8)
axes[1].loglog(nprocs_all, t_fft_late, 'C1.--', zorder=7)
# 37
t_fft_37          = np.array([info_37['data'][i].t_fft          for i in time_steps])
t_longrange_37    = np.array([info_37['data'][i].t_longrange    for i in time_steps])
t_shortrange_37   = np.array([info_37['data'][i].t_shortrange   for i in time_steps])
load_imbalance_37 = np.array([info_37['data'][i].load_imbalance for i in time_steps])
t_fft_early_37        = mean8(t_fft_37       , n_steps=len(time_steps))[ 0 + 8]
t_longrange_early_37  = mean8(t_longrange_37 , n_steps=len(time_steps))[ 0 + 8]
t_shortrange_early_37 = mean8(t_shortrange_37, n_steps=len(time_steps))[ 0 + 8]
t_fft_late_37         = mean8(t_fft_37       , n_steps=len(time_steps))[-1 - 8]
t_longrange_late_37   = mean8(t_longrange_37 , n_steps=len(time_steps))[-1 - 8]
t_shortrange_late_37  = mean8(t_shortrange_37, n_steps=len(time_steps))[-1 - 8]
axes[0].loglog(37, t_longrange_early_37 + t_shortrange_early_37, 'C3.-')
axes[0].loglog(37, t_shortrange_early_37, 'C0.-', zorder=9)
axes[0].loglog(37, t_longrange_early_37, 'C1.-', zorder=8)
axes[0].loglog(37, t_fft_early_37, 'C1.--', zorder=7)
axes[1].loglog(37, t_longrange_late_37 + t_shortrange_late_37, 'C3.-', zorder=10)
axes[1].loglog(37, t_shortrange_late_37, 'C0.-', zorder=9)
axes[1].loglog(37, t_longrange_late_37, 'C1.-', zorder=8)
axes[1].loglog(37, t_fft_late_37, 'C1.--', zorder=7)

# Legend
axes[0].loglog(8, 1, 'C3-',  label='total')
axes[0].loglog(8, 1, 'C0-',  label='short-range')
axes[0].loglog(8, 1, 'C1-',  label='long-range')
axes[0].loglog(8, 1, 'C1--', label='FFT')
axes[0].legend(loc='lower left')
# Perfect scaling
y = np.log(t_longrange_early + t_shortrange_early)
x = np.log(nprocs_all)
a = (y[1] - y[0])/(x[1] - x[0])
b = y[0] - a*x[0]
#axes[0].loglog(nprocs_all, np.exp(a*np.log(nprocs_all) + b), 'k:', zorder=0)  # Which to use?
axes[0].loglog(nprocs_all, 1/np.array(nprocs_all)*(nprocs_all[0]*(t_longrange_early[0] + t_shortrange_early[0])), 'k:', zorder=0)
y = np.log(t_longrange_late + t_shortrange_late)
x = np.log(nprocs_all)
a = (y[1] - y[0])/(x[1] - x[0])
b = y[0] - a*x[0]
#axes[1].loglog(nprocs_all, np.exp(a*np.log(nprocs_all) + b), 'k:', zorder=0)  # Which to use?
axes[1].loglog(nprocs_all, 1/np.array(nprocs_all)*(nprocs_all[0]*(t_longrange_late[0] + t_shortrange_late[0])), 'k:', zorder=0)

# Other
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position('right')
for ax in axes[:2]:
    ax.set_xticks(nprocs_all)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_xlim([nprocs_all[0], nprocs_all[-1]])
axes[0].text(0.95, 0.9, r'$z \sim 99$', va='center', ha='right', transform=axes[0].transAxes)
axes[1].text(0.95, 0.9, r'$z \sim 0$', va='center', ha='right', transform=axes[1].transAxes)
for i, ax in enumerate(axes[:2]):
    #ax.set_xlabel(r'$n_{\mathrm{p}}$')
    #ax.text(i + 0.087*(i*2 - 1), -0.053, r'$n_{\mathrm{p}}$',
    #    transform=ax.transAxes, ha='center', va='center')
    ax.text(0.5, -0.163, r'$n_{\mathrm{p}}$', va='center', ha='center', transform=axes[i].transAxes, clip_on=False, zorder=np.inf)

    ax.set_ylabel(r'computation time per step [s]')
xticklabels = [str(nprocs) for nprocs in nprocs_all]
#xticklabels[-1] = rf'$..\text{{\normalsize$\sfrac{{{nprocs_all[-1]}}}{{{nprocs_all[0]}}}$}}$'
#axes[0].fill(
#    [0.9, 0.98, 0.98, 0.9, 0.9],
#    [-0.12, -0.12, -0.095, -0.095, -0.12],
#    'w', ec=None, transform=axes[0].transAxes, zorder=+np.inf, clip_on=False,
#)

xticklabels[-1] = ''

axes[0].text(1.00, -0.094 - 0.11*0.69, nprocs_all[-1], color='k', ha='center', rotation=0, rotation_mode='anchor', transform=axes[0].transAxes, zorder=1e+100, clip_on=False)

axes[1].fill(
    [0.97, 1.002, 1.002, 0.97, 0.97],
    [-0.07, -0.07, -0.098, -0.098, -0.07],
    'w', ec=None, transform=axes[0].transAxes, zorder=+np.inf, clip_on=False,
)

axes[0].set_xticklabels(xticklabels)
xticklabels = [str(nprocs) for nprocs in nprocs_all]
#xticklabels[0] = ''
xticklabels[0] = '..1'
axes[1].set_xticklabels(xticklabels)
# Lower plot
c = -1
for nprocs, info in infos_time.items():
    if nprocs not in (1, 4, 16, 64, 256, 1024):      
        continue
    c += 1
    color = f'C{c}'
    axes[2].semilogy(
        time_steps,
        mean8([info['data'][i].t_longrange for i in time_steps], n_steps=len(time_steps)),
        '--',
        color=color,
    )
    # Short-range with load imbalance
    t_shortrange = np.array([info['data'][i].t_shortrange for i in time_steps])
    load_imbalance = np.array([info['data'][i].load_imbalance for i in time_steps])
    t_shortrange_std = np.array([
        np.std(t_shortrange[i]*(1 + load_imbalance[i, :])) for i in time_steps
    ])
    axes[2].fill_between(
        time_steps,
        mean8(t_shortrange + t_shortrange_std, n_steps=len(time_steps)),
        mean8(t_shortrange - t_shortrange_std, n_steps=len(time_steps)),
        color=color,
        alpha=0.5,
        edgecolor=None,
    )
    axes[2].semilogy(
        time_steps,
        mean8(t_shortrange, n_steps=len(time_steps)),
        '-',
        color=color,
        label=rf'$n_{{\mathrm{{p}}}} = {nprocs}$',
    )
# Convert x axis to redshift, keeping it linear in time steps
axes[2].set_xlim(time_steps[0], time_steps[-1])
zticks = [99, 40, 20, 10, 5, 3, 2, 1, 0.5, 0]
z_timesteps = np.array([
    1/infos_time[8]['data'][i].scale_factor - 1
    for i in time_steps
])
xticks = [
    np.interp(1/ztick, 1/z_timesteps, time_steps)
    if ztick > 0 else time_steps[-1]
    for ztick in zticks
]
axes[2].set_xticks(xticks)
axes[2].set_xticklabels([str(ztick) for ztick in zticks])
# Other
axes[2].set_xlabel('$z$')
axes[2].set_ylabel('computation time per step [s]')
legend1 = axes[2].legend(loc='upper left', framealpha=0.6)
# Fake legend
legend2 = axes[2].legend(
    (
          axes[2].plot(1, 1, '-' , color='grey')
        + axes[2].plot(1, 1, '--', color='grey')
    ),
    (r'short-range', r'long-range'),
    loc=(0.1718, 0.786), framealpha=0.6,

)
axes[2].add_artist(legend1)
axes[2].add_artist(legend2)
# Remove small legend bits
axes[0].plot([0.108]*2, [0.06, 0.1], 'w', transform=axes[0].transAxes, zorder=np.inf, alpha=1)
axes[2].plot([0.2155]*2, [0.82, 0.87], 'w', transform=axes[2].transAxes, zorder=np.inf, alpha=1)

axes[2].set_ylim(0.234, 1.9e+3)

# Save
fig.subplots_adjust(wspace=0, hspace=0.23)
cropsave(fig, '../figure/strong.pdf')  # no tight_layout() or bbox_inches()

