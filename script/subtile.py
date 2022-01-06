import collections
import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import scipy.optimize
from helper import load, mean8, cropsave, grendel_dir



"""
SUBTILE POPULATION

N = 512³, nprocs = 64.
Two panels (left & right, spanning two cols).
Left: Subtile population at z = 0 for various box sizes.
Right: Subtile population over time for a particular box size.
"""



textwidth = 504  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 2.84
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
output_dir = f'{grendel_dir}/powerspec/boxsize_scaling/64'
infos_boxsize_time = load(output_dir, 'time', check_spectra=0, cache_assume_uptodate=cache_assume_uptodate)

boxsize_max = 1024
infos_boxsize_time = {k: v for k, v in infos_boxsize_time.items() if k <= boxsize_max}

# Plot
fig, axes = plt.subplots(1, 2, figsize=(width, height))
boxsizes = np.array(sorted(list(infos_boxsize_time.keys())))
N = 512**3
nprocs = 64
subtiling_populations = [np.array(infos_boxsize_time[boxsize]['data'][-3].subtiling) for boxsize in boxsizes]
subtiling_populations_padded = []
subtiling_max = 10
for i in range(1, subtiling_max + 1):
    tmp = []
    for subtiling_population in subtiling_populations:
        tmp.append(collections.Counter(subtiling_population)[i])
        #if i < len(subtiling_population):
        #    tmp.append(subtiling_population[i])
        #else:
        #    tmp.append(0)
    subtiling_populations_padded.append(tmp)
labels = [f'${boxsize}$' for boxsize in boxsizes]
width = 0.24
i = subtiling_max + 1
running = np.zeros(len(boxsizes))
ax = axes[0]
for subtiling_population_padded in reversed(subtiling_populations_padded):
    i -= 1
    Y = np.array(subtiling_population_padded)/nprocs
    ax.bar(boxsizes, Y, width*boxsizes, bottom=running, label=rf'${i}\times {i}\times {i}$', color=f'C{i-1}')
    running += Y
#ax.set_yscale('log')
ymin = 0
ax.set_ylim(ymin, 1)
ax.set_xscale('log')
ax.invert_xaxis()
#ax.legend()
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles[::-1], labels[::-1], loc='center right')
ax.set_xlabel('$L_{\mathrm{box}}\; [\mathrm{Mpc}/h]$')
ax.set_ylabel('subtile distribution')
ax_bot = ax
ax_top = ax.twiny()
frac = 1.125
ax_bot.set_xlim(boxsizes[0]/frac, boxsizes[-1]*frac)
ax_bot.set_xticks(boxsizes)
ax_bot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_bot.tick_params(axis='x', which='minor', bottom=False)
ax_bot.invert_xaxis()

boxsizes_axis = []
boxsize = boxsize_max
while boxsize > 48:
    boxsizes_axis.append(boxsize)
    boxsizes_axis.append(boxsize//4*3)
    boxsize //= 2
boxsizes_axis.pop()
boxsizes_axis.pop()
boxsizes_axis = np.array(boxsizes_axis[::-1])
#ax_bot.set_xlim(boxsizes_axis[0], boxsizes_axis[-1])
ax_bot.set_xticks(boxsizes_axis)
ax_bot.set_xticklabels([f'${b}$' for b in boxsizes_axis],
    rotation=34, ha='right', rotation_mode='anchor')

k_Nyquist = 2*np.pi/boxsizes[::-1]*(512/2)
xticks = np.log(k_Nyquist)
xticks -= xticks[0]
xticks = [xtick/xticks[-1] for xtick in xticks]
xticklabels = [f'${k:.3g}$' for k in k_Nyquist]
ax_top.set_xticks(xticks)
#ax_top.set_xticklabels(xticklabels)
ax_top.set_xticklabels(xticklabels, rotation=34, ha='left', rotation_mode='anchor')
frac *= 0.98  # unexplained
ax_top.set_xlim(0 - np.log10(frac), 1 + np.log10(frac))
ax_top.set_xlabel('$k_{\mathrm{Nyquist}}\; [h/\mathrm{Mpc}]$')

# Right panel
box = 128
ax = axes[1]
time_steps = np.arange(len(infos_boxsize_time[box]['data']))

#ax.semilogy(
#    time_steps,
#    np.random.random(len(time_steps)),
#    '-',
#)
time_step_info = infos_boxsize_time[box]['data']
highest_subtiling = subtiling_max
population = {i: np.zeros(len(time_step_info), dtype=int) for i in range(1, highest_subtiling + 1)}
for step, d in enumerate(time_step_info):
    counter = collections.Counter(d.subtiling)
    for i in range(1, highest_subtiling + 1):
        population[i][step] += counter[i]
#    rp = list(d.subtiling)
#    for i, pop in enumerate(rp, 1):
#        if i > highest_subtiling:
#            i = highest_subtiling
#        population[i][step] += pop

x = [d.time_step for d in time_step_info]
population_fraction = list(population.values())
NN = np.sum([_[0] for _ in population_fraction])
population_fraction_new = []
for arr in population_fraction:
    l = 16
    arr = list(arr) + [arr[-1]]*l
    meanarr = mean8(np.array(arr)/NN, period=16, n_steps=len(time_steps)+l)
    meanarr = meanarr[:-l]
    population_fraction_new.append(
        #np.array(_)/NN
        meanarr
    )
population_fraction = population_fraction_new
ax.stackplot(
    x[len(x) - len(population_fraction[0]):],
    population_fraction[::-1],
    labels=[rf'${i}\times {i}\times {i}$' for i in range(1, highest_subtiling + 1)][::-1],
    colors=[
        np.asarray(matplotlib.colors.ColorConverter().to_rgb(f'C{i-1}'), dtype=float)
        for i in range(1, highest_subtiling + 1)
    ][::-1],
)
#ax.set_yscale('log')
#ax.set_ylim(1e-5, 1)

# Convert x axis to redshift, keeping it linear in time steps
first_step_to_show = 12
ax.set_xlim(time_steps[first_step_to_show], time_steps[-1])
zticks = [99, 40, 20, 10, 5, 3, 2, 1, 0.5, 0]
zticks = zticks[2:]
z_timesteps = np.array([
    1/infos_boxsize_time[box]['data'][i].scale_factor - 1
    for i in time_steps
])
xticks = [
    np.interp(1/ztick, 1/z_timesteps, time_steps)
    if ztick > 0 else time_steps[-1]
    for ztick in zticks
]
ax.set_xticks(xticks)
ax.set_xticklabels([str(ztick) for ztick in zticks])
ax.set_xlabel('$z$')
# Legend
ax.legend()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', framealpha=0.6)
# "Inout" via plotted lines
for y in (0.2, 0.4, 0.6, 0.8):
    ax.plot([9.4, 14.6], [y]*2, 'k', clip_on=False, zorder=np.inf, lw=0.75)
#ax.tick_params('y', direction='inout', which='both')
ax.yaxis.tick_right()
ax.set_ylim(ymin, 1)
#ax.set_yscale('log')
#ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])
#ax.set_yticklabels()
ax.set_ylabel('subtile distribution')
ax.yaxis.set_label_position('right')
#ax.text(0.5, 0.5, rf'$L_{{\text{{box}}}} = {box}\,\mathrm{{Mpc}}/h$', transform=ax.transAxes)
ax.set_title(rf'$L_{{\text{{box}}}} = \SI{{{box}}}{{Mpc}}/h$', fontsize=fontsize)
#for ax in axes:
#    ax.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
#    ax.set_yticklabels([r'$1$', r'$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'])

# Save
fig.subplots_adjust(wspace=0, hspace=0)
cropsave(fig, '../figure/subtile.pdf')  # no tight_layout() or bbox_inches()

