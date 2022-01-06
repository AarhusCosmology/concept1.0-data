import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import scipy.optimize
from helper import load, mean8, cropsave, grendel_dir



"""
RUNGPOPULATION

N = 512³, nprocs = 64.
Two panels (left & right, spanning two cols).
Left: Rung population at z = 0 for various box sizes.
Right: Rung population over time for a particular box size.
"""



textwidth = 504  # mnras: 240 (single-column), 504 (both columns)
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

# Plot
fig, axes = plt.subplots(1, 2, figsize=(width, height))
boxsizes = np.array(sorted(list(infos_boxsize_time.keys())))
N = 512**3
rung_populations = [np.array(infos_boxsize_time[boxsize]['data'][-3].rung_population)/N for boxsize in boxsizes]
rung_populations_padded = []
rung_max = 5  # actually 5, but only for boxsize = 1024 Mpc/h
for i in range(rung_max + 1):
    tmp = []
    for rung_population in rung_populations:
        if i < len(rung_population):
            tmp.append(rung_population[i])
        else:
            tmp.append(0)
    rung_populations_padded.append(tmp)
labels = [f'${boxsize}$' for boxsize in boxsizes]
width = 0.24
i = rung_max + 1
running = np.zeros(len(boxsizes))
ax = axes[0]
for rung_population_padded in reversed(rung_populations_padded):
    i -= 1
    ax.bar(boxsizes, rung_population_padded, width*boxsizes, bottom=running, label=f'rung {i}',
        color=f'C{i}')
    running += np.array(rung_population_padded)
ax.set_yscale('log')
ymin = 2.5e-7
ax.set_ylim(ymin, 1)
ax.set_xscale('log')
ax.invert_xaxis()
#ax.legend()
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles[::-1], labels[::-1], loc='center right')
ax.set_xlabel('$L_{\mathrm{box}}\; [\mathrm{Mpc}/h]$')
ax.set_ylabel('rung distribution')
ax_bot = ax
ax_top = ax.twiny()
frac = 1.125
ax_bot.set_xlim(boxsizes[0]/frac, boxsizes[-1]*frac)
ax_bot.set_xticks(boxsizes)
ax_bot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_bot.tick_params(axis='x', which='minor', bottom=False)
ax_bot.invert_xaxis()

boxsizes_axis = []
boxsize = 4096
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
box = 1024
ax = axes[1]
time_steps = np.arange(len(infos_boxsize_time[box]['data']))
time_step_info = infos_boxsize_time[box]['data']
highest_rung = rung_max
population = {i: np.zeros(len(time_step_info), dtype=int) for i in range(highest_rung + 1)}
for step, d in enumerate(time_step_info):
    rp = list(d.rung_population)
    for i, pop in enumerate(rp):
        if i > highest_rung:
            i = highest_rung
        population[i][step] += pop
x = [d.time_step for d in time_step_info]
population_fraction = list(population.values())
NN = np.sum([_[0] for _ in population_fraction])
population_fraction = [
    #np.array(_)/NN
    mean8(np.array(_)/NN, n_steps=len(time_steps))
    for _ in population_fraction
]
ax.stackplot(
    x,
    population_fraction[::-1],
    labels=[f'rung {i}' for i in range(highest_rung + 1)][::-1],
    colors=[
        np.asarray(matplotlib.colors.ColorConverter().to_rgb(f'C{i}'), dtype=float)
        for i in range(highest_rung + 1)
    ][::-1],
)
#ax.set_yscale('log')
#ax.set_ylim(1e-5, 1)

# Convert x axis to redshift, keeping it linear in time steps
first_step_to_show = 22
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
ax.legend(framealpha=0.6)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', framealpha=0.6)
# "Inout" via plotted lines
for y in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    ax.plot([21.3, 22.7], [y]*2, 'k', clip_on=False, zorder=np.inf, lw=0.75)
ax.yaxis.tick_right()
ax.set_ylim(ymin, 1)
ax.set_yscale('log')
#ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])
#ax.set_yticklabels()
ax.set_ylabel('rung distribution')
ax.yaxis.set_label_position('right')
#ax.text(0.5, 0.5, rf'$L_{{\text{{box}}}} = {box}\,\mathrm{{Mpc}}/h$', transform=ax.transAxes)
ax.set_title(rf'$L_{{\text{{box}}}} = \SI{{{box}}}{{Mpc}}/h$', fontsize=fontsize)
for ax in axes:
    ax.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    ax.set_yticklabels([r'$1$', r'$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'])

# Save
fig.subplots_adjust(wspace=0, hspace=0)
cropsave(fig, '../figure/rung.pdf')  # no tight_layout() or bbox_inches()

