import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import scipy.optimize
from helper import load, mean8, cropsave, grendel_dir



"""
MEMORY SCALING

Memory usage as function of problem size,
at z = 0.
This uses boxsize = 2*cbrt(N)*Mpc/h.
"""



textwidth = 240  # mnras: 240 (single-column), 504 (both columns)
width = textwidth/72.27
height = 2.483
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
output_dir = f'{grendel_dir}/powerspec/weak_scaling'
infos_weak_mem  = load(output_dir, 'mem',  check_spectra=0, cache_assume_uptodate=cache_assume_uptodate)

# The semi-analytical mem estimate
def get_mem(size, nprocs, N_rungs=10, nghosts=2, interlace=True):
    """
    Possible sources of extra memory:
    - mem_load       # to test: reduce buffer size by a lot
    - mem_exchange   # to test: reduce buffer size by a lot
    - mem_ghost      # to test: run with no forces at all (also disables a lot of other things)
    """
    N = size**3
    mem_tot = 0
    # Base memory (code, lib, persistent objects)
    mem_base = 160*2**20*1
    mem_base += 35*2**20*(nprocs - 1)
    mem_tot += mem_base    
    # Read in of GADGET snapshot
    mem_load = (2**23)*(4/8)*nprocs
    mem_tot += mem_load
    # Particle pos, mom, Δmom, rung_indices, rung_indices_jumped and the tmp rung buffer
    fac = 1 + 0.3*(nprocs > 1)
    mem_particles = N*((3 + 3 + 3)*8 + (1 + 1 + 1)*1)*fac
    mem_tot += mem_particles
    # Particle exchange
    fac = 10  # "unlucky factor" accounting for a number of recvs with no sends,
             # growing local component data.
             # Can also be thought of as taking (particle data) load imbalance into account.
    mem_exchange = (
        4*nprocs*8  # global buffers
        + (nprocs > 1)*2**17*((3 + 3 + 3)*8 + (1 + 1)*1)*nprocs*fac  # component resizing
    )
    mem_tot += mem_exchange
    # PM grid/slab. The *3 is because we have domain grid, slab and force grid.
    mem_grid = (2*size)**3*8
    mem_pm = mem_grid*3
    mem_tot += mem_pm
    # PM grid ghost layers. The *2 is because we have domain and force grid.
    mem_ghost = (((2*size)/np.cbrt(nprocs) + 2*nghosts)**3*nprocs - (2*size)**3)*8
    mem_pm_ghosts = mem_ghost*2
    mem_tot += mem_pm_ghosts
    # Domain/slab decompositions
    mem_decompositions = 2**20*(1 + 1)*8*nprocs
    mem_tot += mem_decompositions
    # Power spectrum grid/slab. If not using interlacing, the PM grid/slab is reused.
    mem_powerspec = interlace*(mem_grid + mem_ghost)
    mem_tot += mem_powerspec
    # Primary tiling
    size_tiling = (2*size)/(4.5*1.25)
    mem_tiling = (size_tiling**3*(7 + 3*N_rungs) + 3*N)*8 + size_tiling**3*1
    mem_tot += mem_tiling
    # Additional tiling used for the supplier component
    mem_tiling_comm = (nprocs > 1)*(
        (size_tiling**3*(7 + 3*N_rungs) + N*(1 - (size_tiling - 2)**3/(size_tiling**3)))*8 + size_tiling**3*1
    )
    mem_tot += mem_tiling_comm
    # Pre-calculated tile pairings. I can't explain the weird factor ~5.
    weird_fac = 5
    mem_tile_pairs = weird_fac*size_tiling**3*(3**3/2 + 1)*8
    mem_tot += mem_tile_pairs
    # Total memory in bytes
    return mem_tot

#x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#x = np.array(x)
#estimate = np.array([get_mem(size=512, nprocs=nprocs, interlace=False) for nprocs in x])/2**30
#plt.plot(x, estimate + 0, 'r--')
def cheeky_mem_estimate(N, nprocs):
    a = (
        + 0*4  # extra mem from nowhere, for N = 512³, boxsize = 1024*Mpc/h
        + 0.14*nprocs
        + 3.2e-07*N
        + 7e-7*nprocs**(1/3)*N**(2/3)  # ghost zones (not very important)
    )
    return a
def cheeky_mem_estimate2(N, nprocs):
    nghosts = 2
    size = N**(1/3)
    N_rungs = 10
    total_mem = (
        + 1*(
            #+ (160 - 35)*2**20  # base
            #+ 8*177.978515625  # tiling comm
            #131073423.828125*0.98
            0.119
        )
        + nprocs*(
            #+ 35*2**20       # base
            #+ (2**23)*(4/8)  # read in of snapshot
            #+ 4*8 + 2**17*((3 + 3 + 3)*8 + (1 + 1)*1)*10  # exchange
            #+ 2**20*(1 + 1)*8  # decomp
            #+ 128*8  # ghosts
            #154666016
            0.144
        )
        + N*(
            #+ ((3 + 3 + 3)*8 + (1 + 1 + 1)*1)*1.3  # particle data
            #+ 8*8*3  # PM
            #+ 37.35  # primary tiling
            #+ 26.07056  # pre-calculated tile pairings
            #+ 13.3499  # tiling comm
            #366.27046*1.007
            3.46e-7
        )
        #+ N**(2/3)*(
        #    + 768*nprocs**(1/3)  # ghosts
        #    + 135  # tiling comm
        #)
        #+ N**(1/3)*(
        #    + 1536*nprocs**(2/3)  # ghosts
            #- 759.375  # tiling comm
        #)
    )
    return total_mem

# Plot
fig = plt.figure(figsize=(width, height))
x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
nprocs_all = x
Ns = np.array(nprocs_all)/1024*2048**3
sizes = np.asarray(np.round((2*np.round(np.cbrt(Ns)/2)*2)/np.array(nprocs_all))*np.array(nprocs_all)/2, dtype=int)
y = [0]*len(x)
y_std = [0]*len(x)
for nprocs, info in infos_weak_mem.items():
    mem_tot = info['mem']['PSS'].sum(axis=0)
    mem_std = info['mem']['PSS'].std(axis=0)
    y    [x.index(nprocs)] = mem_tot[-4]/2**30
    y_std[x.index(nprocs)] = mem_std[-4]/2**30*nprocs
x     = np.array(x)
y     = np.array(y)
y_std = np.array(y_std)
plt.errorbar(sizes**3, y, yerr=y_std, fmt=f'C0.', zorder=10)
plt.xlabel('$n_{\mathrm{p}}$')
plt.ylabel(r'total memory $M\; [\text{GB}]$')
#plt.title('$z = 0$')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xticks(sizes**3)
# Perfect scaling
y_perfect = sizes**3/sizes[1]**3*y[1]
plt.plot(sizes**3, y_perfect, 'k:', zorder=100)
# Estimate
estimate = np.array([get_mem(size=size, nprocs=nprocs, interlace=False) for nprocs, size in zip(nprocs_all, sizes)])/2**30
#plt.plot(sizes**3, estimate, 'r--')
# Cheeky estimate
y = np.array([cheeky_mem_estimate(size**3, nprocs) for nprocs, size in zip(nprocs_all, sizes)])
#plt.plot(sizes**3, y, 'g--')
# Cheeky estimate 2
y = np.array([cheeky_mem_estimate2(size**3, nprocs) for nprocs, size in zip(nprocs_all, sizes)])
plt.plot(sizes**3, y, 'C0-', zorder=5)

# Top x axis
ax_bot = plt.gca()
ax_top = plt.gca().twiny()
ax_bot.set_xlim(sizes[0]**3, sizes[-1]**3)
ax_bot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_bot.tick_params(axis='x', which='minor', bottom=False)
ax_bot.set_xticklabels([f'${nprocs}$' for nprocs in nprocs_all])
xticklabels = [rf'${size}^3$' for size in sizes]
xticks = np.log(ax_bot.get_xticks())
xticks -= xticks[0]
xticks *= 1/xticks[-1]
ax_top.set_xticks(xticks)
ax_top.set_xticklabels(xticklabels, rotation=34, ha='left', rotation_mode='anchor')
ax_top.set_xlabel(r'$N$')

# Save
cropsave(fig, '../figure/mem.pdf')  # no tight_layout() or bbox_inches()

