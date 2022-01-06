import collections
import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import scipy.optimize
from helper import load, mean8, cropsave, get_factor_after_symplectifying, grendel_dir



"""
SUBTILE REFINEMENT

N = 512³, nprocs = 64. Two panels.
Left panel: Total short-range computation time
(plus time spent on refinements, i.e. total time per
step minus long-range) as function of choice of subtile
refinement.
Right panel: Time per step (as above) as function of z,
for some (if not all) of the sims used for the left panel.
"""



textwidth = 504  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 2.78  # we would like it smaller, but that cuts off the ylabels
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
output_dir = f'{grendel_dir}/powerspec/subtiling_scaling'
box = [256, 192][1]
infos = load(output_dir, f'box{box}', check_spectra=0, cache_assume_uptodate=cache_assume_uptodate)

# Mapping:
#   100: dynamic (16)
#   101: dynamic (8)
#   102: dynamic (24)
#   103: dynamic (32)
if box == 192:
    chosen = 300
    infos[4] = infos[401]
else:
    chosen = 100
period = {
    100: 16, 101:8, 102: 24, 103: 32,
    201: 8, 202: 8, 203: 8, 204: 8, 205:8,
    300: 16, 400: 16, 500: 16,
}[chosen]
keys = np.array([int(k) for k in infos.keys() if int(k) < 100 or int(k) == chosen])
infos = {int(k): v for k, v in infos.items() if int(k) in keys}
infos[0] = infos[chosen]
infos.pop(chosen)
keys = np.array(list(infos.keys()))
#keys = np.array(list(keys) + [1])
#infos[1] = infos[2]
keys.sort()
infos = {k: infos[k] for k in keys}

shortrange_times = {}
for refinement in infos.keys():
    shortrange_times[refinement] = infos[refinement]['computation_times'] - np.array([step.t_longrange for step in infos[refinement]['data']])
shortrange_times_total = {key: np.sum(arr) for key, arr in shortrange_times.items()}
keys = np.array(list(infos.keys()))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(width, height))
sympletic_performancehit = get_factor_after_symplectifying()
N = 512**3
nprocs = 64
width = 0.8
y = np.array(list(shortrange_times_total.values()))/60**2
c = -1
color_auto = [0.20]*3
for key in keys:
    if key == 0:
        x = keys[-1] + 1
        color = color_auto
    else:
        x = key
        c += 1
        color = f'C{c}'
    axes[0].bar(x, sympletic_performancehit*y[key], width, color=color)
axes[0].plot((-2, keys[-1]+4), [0.99798*sympletic_performancehit*y[0]]*2, 'k:')
axes[0].set_xlim(1 - width/2, keys[-1] + 1 + width/2)
xticks = np.arange(keys[1], keys[-1]+2)
xticklabels = [rf'${xtick}\times {xtick}\times {xtick}$' for xtick in xticks]
xticklabels[-1] = 'dynamic'
axes[0].set_xticks(xticks)
axes[0].set_xticklabels(xticklabels, rotation=34, ha='right', rotation_mode='anchor')
ymax = sympletic_performancehit*np.max(y)
ymin = sympletic_performancehit*np.min(y)
axes[0].set_ylim(
    20, #ymin - (ymax - ymin)*0.7,
    ymax + (ymax - ymin)*0.05,
)
axes[0].set_xlabel(r'subtile decomposition')
axes[0].set_ylabel(r'total short-range computation time [hr]')

# Right panel
time_steps = np.arange(len(infos[1]['data']))
c = -1
for key in keys:
    zorder = 10
    if key == 0:
        zorder = 100
        color = color_auto
    else:
        c += 1
        color = f'C{c}'

    info = infos[key]
    # Short-range with load imbalance
    t_shortrange = np.array([info['data'][i].t_shortrange for i in time_steps])
    load_imbalance = np.array([info['data'][i].load_imbalance for i in time_steps])
    t_shortrange_std = np.array([
        np.std(t_shortrange[i]*(1 + load_imbalance[i, :])) for i in time_steps
    ])
    #axes[1].fill_between(
    #    time_steps[2:],
    #    #(shortrange_times[key] + t_shortrange_std)[2:],
    #    #(shortrange_times[key] - t_shortrange_std)[2:],
    #    mean8(shortrange_times[key] + t_shortrange_std, n_steps=len(time_steps)),
    #    mean8(shortrange_times[key] - t_shortrange_std, n_steps=len(time_steps)),
    #    color=color,
    #    alpha=(0.3 if key == 0 else 0.2),
    #    edgecolor=None,
    #)
    #axes[1].semilogy(
    #    time_steps[2:],
    #    #shortrange_times[key][2:],
    #    mean8(shortrange_times[key], n_steps=len(time_steps)),
    #    '-',
    #    color=color,
    #    #label=rf'$n_{{\mathrm{{p}}}} = {nprocs}$',
    #    zorder=zorder,
    #)
    y = mean8(shortrange_times[key]/shortrange_times[1], period=period, n_steps=len(time_steps))
    #y = shortrange_times[key]/shortrange_times[1]
    axes[1].plot(
        time_steps[len(time_steps) - len(y):],
        #shortrange_times[key][2:],
        y - 1,
        '-',
        color=color,
        #label=rf'$n_{{\mathrm{{p}}}} = {nprocs}$',
        zorder=zorder,
    )

# Convert x axis to redshift, keeping it linear in time steps
zticks = [99, 40, 20, 10, 5, 3, 2, 1, 0.5, 0]
z_timesteps = np.array([
    1/infos[1]['data'][i].scale_factor - 1
    for i in time_steps
])
xticks = [
    np.interp(1/ztick, 1/z_timesteps, time_steps)
    if ztick > 0 else time_steps[-1]
    for ztick in zticks
]
axes[1].set_xticks(xticks)
axes[1].set_xticklabels([str(ztick) for ztick in zticks])
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position('right')
axes[1].set_xlim(time_steps[57], time_steps[-1])
if box == 256:
    axes[1].set_ylim(0.8 - 1, 1.25 - 1)
elif box == 192:
    axes[1].set_ylim(0.73 - 1, 1.25 - 1)
    axes[1].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    axes[1].set_yticklabels([r'$\SI{-20}{\percent}$', r'$\SI{-10}{\percent}$', r'$0$', r'$\SI{10}{\percent}$', r'$\SI{20}{\percent}$'])

# Other
axes[1].set_xlabel('$z$')
axes[1].set_ylabel(r'short-range computation time per step' + '\n' + r'relative to subtile decomposition $1\times 1\times 1$')

# Save
fig.subplots_adjust(wspace=0, hspace=0)
cropsave(fig, '../figure/subtile_refinement.pdf')  # no tight_layout() or bbox_inches()

