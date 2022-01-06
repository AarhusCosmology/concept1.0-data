import collections, os, sys
import numpy as np
import scipy.integrate
from scipy.special import erf, erfc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import noise

matplotlib.rcParams['hatch.linewidth'] = 0.7

fontsize = 11/1.4
latex_preamble = r'''
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{mathtools}
    \usepackage{siunitx}
    \usepackage{slantsc}
    \usepackage{graphicx}
    \usepackage{abraces}
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



# Specs
Mpc = 1
boxsize = 512*Mpc
domain_subdivisions = (2, 3)
gridsize = 54
shortrange_scale = 1.25*boxsize/gridsize
shortrange_range = 4.5*shortrange_scale
shortrange_tilesize = shortrange_range
tiling = np.asarray(
    (boxsize/np.asarray(domain_subdivisions))/shortrange_tilesize*(1 + 1e-16),
    dtype=int,
)
subtilings = [(2, 1), (1, 1), (3, 2), (1, 1), (2, 2), (4, 3)]

# Initialize figure
fig, ax = plt.subplots()
# Draw box
lw_box = 1.5
plt.plot([0, boxsize, boxsize, 0, 0], [0, 0, boxsize, boxsize, 0],
    'k-', lw=lw_box, zorder=300)
# Draw domains
for dim in range(2):
    for i in range(1, domain_subdivisions[dim]):
        x, y = [0, boxsize], [boxsize/domain_subdivisions[dim]*i]*2
        if dim == 1:
            x, y = y, x
        plt.plot(x, y, 'k-', lw=1.3, zorder=90)
# Draw PM grid
for i in range(1, gridsize):
    plt.plot([0, boxsize], [boxsize/gridsize*i]*2, '-', color=[0.89]*3, lw=0.5, zorder=-100)
    plt.plot([boxsize/gridsize*i]*2, [0, boxsize], '-', color=[0.89]*3, lw=0.5, zorder=-100)
# Draw tiling
for dim in range(2):
    for i in range(1, tiling[dim]*domain_subdivisions[dim]):
        x, y = [0, boxsize], [boxsize/domain_subdivisions[dim]/tiling[dim]*i]*2
        if dim == 1:
            x, y = y, x
        plt.plot(x, y, 'C5-', lw=0.95, zorder=70)
# Draw subtilings
Subtile = collections.namedtuple('Subtile', ['x', 'y', 'width', 'height'])
subtiles = []
domain_size_x = boxsize/domain_subdivisions[1]
domain_size_y = boxsize/domain_subdivisions[0]
tile_size_x = domain_size_x/tiling[1]
tile_size_y = domain_size_y/tiling[0]
n = -1
for i in range(domain_subdivisions[0]):
    for j in range(domain_subdivisions[1]):
        n += 1
        subtiling = subtilings[n]
        domain_start_x = domain_size_x*j
        domain_start_y = domain_size_y*i
        for ii in range(tiling[0]):
            for jj in range(tiling[1]):
                tile_start_x = domain_start_x + tile_size_x*jj
                tile_start_y = domain_start_y + tile_size_y*ii
                for iii in range(subtiling[0]):
                    for jjj in range(subtiling[1]):
                        subtile = Subtile(
                            tile_start_x + tile_size_x/subtiling[1]*jjj,
                            tile_start_y + tile_size_y/subtiling[0]*iii,
                            tile_size_x/subtiling[1],
                            tile_size_y/subtiling[0],
                        )
                        subtiles.append(subtile)
                for iii in range(1, subtiling[0]):
                    plt.plot(
                        [tile_start_x, tile_start_x + tile_size_x],
                        [tile_start_y + tile_size_y/subtiling[0]*iii]*2,
                        'C1-', lw=0.6, zorder=60,
                    )
                for jjj in range(1, subtiling[1]):
                    plt.plot(
                        [tile_start_x + tile_size_x/subtiling[1]*jjj]*2,
                        [tile_start_y, tile_start_y + tile_size_y],
                        'C1-', lw=0.6,  zorder=60,
                    )

def generate_subtiles(n):
    subtiling = subtilings[n]
    subtiles = []
    for i in range(domain_subdivisions[0]):
        for j in range(domain_subdivisions[1]):
            domain_start_x = domain_size_x*j
            domain_start_y = domain_size_y*i
            for ii in range(tiling[0]):
                for jj in range(tiling[1]):
                    tile_start_x = domain_start_x + tile_size_x*jj
                    tile_start_y = domain_start_y + tile_size_y*ii
                    for iii in range(subtiling[0]):
                        for jjj in range(subtiling[1]):
                            subtile = Subtile(
                                tile_start_x + tile_size_x/subtiling[1]*jjj,
                                tile_start_y + tile_size_y/subtiling[0]*iii,
                                tile_size_x/subtiling[1],
                                tile_size_y/subtiling[0],
                            )
                            subtiles.append(subtile)
    return subtiles
ghostly_subtiles = [generate_subtiles(n) for n in range(len(subtilings))]

# Draw particles
def get_subtile_dist(subtile1, subtile2):      
    x1 = np.array((subtile1.x, subtile1.x + subtile1.width))
    y1 = np.array((subtile1.y, subtile1.y + subtile1.height))
    x2 = np.array((subtile2.x, subtile2.x + subtile2.width))
    y2 = np.array((subtile2.y, subtile2.y + subtile2.height))
    if subtile1.x - subtile2.x > 0.5*boxsize:
        x2 += boxsize
    elif subtile1.x - subtile2.x < -0.5*boxsize:
        x2 -= boxsize
    if subtile1.y - subtile2.y > 0.5*boxsize:
        y2 += boxsize
    elif subtile1.y - subtile2.y < -0.5*boxsize:
        y2 -= boxsize
    if max(y1) < min(y2):
        # 1 fully below 2
        dy = max(y1) - min(y2)
    elif min(y1) > max(y2):
        # 1 fully above 2
        dy = min(y1) - max(y2)
    else:
        # overlap in y-direction
        dy = 0
    if max(x1) < min(x2):
        # 1 fully to the left of 2
        dx = max(x1) - min(x2)
    elif min(x1) > max(x2):
        # 1 fully to the right of 2
        dx = min(x1) - max(x2)
    else:
        # overlap in x-direction
        dx = 0
    return np.sqrt(dx**2 + dy**2)

theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
N = 8
eps_soft = 0.030*boxsize/np.cbrt(N)
def place_particle(x, y, color='r', hatch='/'*8, place_dot=True):
    # Draw particle with radius equalt to softening length
    if place_dot:
        plt.fill(x + eps_soft*np.cos(theta), y + eps_soft*np.sin(theta), color,
            zorder=304, edgecolor='none')
    X = x + shortrange_range*np.cos(theta)
    Y = y + shortrange_range*np.sin(theta)
    def plot(x, y):
        if len(x) == 0:
            return
        diff_avg = np.mean(np.abs(np.diff(x)))
        index = -1
        for i in range(1, len(x)):
            diffx = abs(x[i] - x[i - 1])
            if diffx > 10*diff_avg:
                index = i
                break
            diffy = abs(y[i] - y[i - 1])
            if diffy > 10*diff_avg:
                index = i
                break
        if index != -1:
            x = np.roll(x, -index)
            y = np.roll(y, -index)
        plt.plot(x, y, '-', color=color, lw=1.3, alpha=0.85, zorder=301)
    mask_ok_x = (0 <= X) & (X < boxsize)
    mask_ok_y = (0 <= Y) & (Y < boxsize)
    mask_left_x = (X < 0)
    mask_right_x = (boxsize <= X)
    mask_down_y = (Y < 0)
    mask_up_y = (boxsize <= Y)
    mask = mask_ok_x & mask_ok_y
    plot(X[mask], Y[mask])
    mask = mask_ok_x & mask_down_y
    plot(X[mask], Y[mask] + boxsize)
    mask = mask_ok_x & mask_up_y
    plot(X[mask], Y[mask] - boxsize)
    mask = mask_left_x & mask_ok_y
    plot(X[mask] + boxsize, Y[mask])
    mask = mask_right_x & mask_ok_y
    plot(X[mask] - boxsize, Y[mask])
    mask = mask_left_x & mask_down_y
    plot(X[mask] + boxsize, Y[mask] + boxsize)
    mask = mask_left_x & mask_up_y
    plot(X[mask] + boxsize, Y[mask] - boxsize)
    mask = mask_right_x & mask_down_y
    plot(X[mask] - boxsize, Y[mask] + boxsize)
    mask = mask_right_x & mask_up_y
    plot(X[mask] - boxsize, Y[mask] - boxsize)
    # Hatch subtiles within which to search
    n = int(x/domain_size_x) + domain_subdivisions[1]*int(y/domain_size_y)
    for subtile in ghostly_subtiles[n]:
        if (
                (subtile.x <= x < subtile.x + subtile.width)
            and (subtile.y <= y < subtile.y + subtile.height)
        ):
            break
    else:
        print(f'Failed to find subtile of particle at ({x}, {y})!', file=sys.stderr)
        sys.exit(1)
    Xs = []
    Ys = []
    for other_subtile in ghostly_subtiles[n]:
        dist = get_subtile_dist(subtile, other_subtile)
        if dist < shortrange_range:
            X = np.array([
                other_subtile.x,
                other_subtile.x + other_subtile.width,
                other_subtile.x + other_subtile.width,
                other_subtile.x,
                other_subtile.x,
            ])
            Y = np.array([
                other_subtile.y,
                other_subtile.y,
                other_subtile.y + other_subtile.height,
                other_subtile.y + other_subtile.height,
                other_subtile.y,
            ])
            if hatch is not None:
                plt.fill(X, Y,
                    color='none', edgecolor=color, zorder=-90, hatch=hatch,
                    fill=False, lw=0, alpha=0.50,
                )
            Xs.append(X)
            Ys.append(Y)
    # Draw boundary of hatched region
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    def draw_hatched_boundary(x, y, X, Y, no=None):
        indices = []
        fac = 1/np.prod(subtilings[n])**1.2
        for q in theta:
            L = 0.5*boxsize
            while L > 0:
                L -= 0.2*boxsize/gridsize
                dist2 = (X - (x + L*np.cos(q)))**2 + (Y - (y + L*np.sin(q)))**2
                if np.min(dist2) < (19.8*boxsize/gridsize*fac)**2:
                    index = np.argmin(dist2)
                    if index not in indices:
                        indices.append(index)
                    break
        indices.append(indices[0])
        indices = np.array(indices)
        X = X[indices]
        Y = Y[indices]
        if no is None:
            no = []
        if isinstance(no, str):
            no = [no]
        no = list(no)
        for no in no:
            nans_to_be_inserted = []
            if no == 'bottom':
                Z = Y
                extrema = np.nanmin(Z)
            elif no == 'top':
                Z = Y
                extrema = np.nanmax(Z)
            elif no == 'left':
                Z = X
                extrema = np.nanmin(Z)
            elif no == 'right':
                Z = X
                extrema = np.nanmax(Z)
            if no is not None:
                reached = False
                for i, zi in enumerate(Z):
                    if not reached:
                        if zi == extrema:
                            reached = True
                        continue
                    if zi == extrema:
                        nans_to_be_inserted.append(i)
                    else:
                        reached = False
            count = 0
            for index in nans_to_be_inserted:
                X = np.insert(X, index + count, np.nan)
                Y = np.insert(Y, index + count, np.nan)
                count += 1
        if hatch is not None:
            plt.plot(X, Y, '--', color=color, lw=0.9, zorder=301, alpha=0.80)
    i = int(x/tile_size_x)
    j = int(y/tile_size_y)
    if 0 < j < domain_subdivisions[0]*tiling[0] - 1 and 0 < i < domain_subdivisions[1]*tiling[1] - 1:
        draw_hatched_boundary(x, y, X, Y)
    elif j == 0 and 0 < i < domain_subdivisions[1]*tiling[1] - 1:
        draw_hatched_boundary(x, y, X, Y, no='bottom')
        draw_hatched_boundary(x, y + boxsize, X, Y, no='top')
    elif i == 0 and 0 < j < domain_subdivisions[0]*tiling[0] - 1:
        draw_hatched_boundary(x, y, X, Y, no='left')
        draw_hatched_boundary(x + boxsize, y, X, Y, no='right')
    elif j == domain_subdivisions[0]*tiling[0] - 1 and 0 < i < domain_subdivisions[1]*tiling[1] - 1:
        draw_hatched_boundary(x, y, X, Y, no='top')
        draw_hatched_boundary(x, y - boxsize, X, Y, no='bottom')
    elif i == domain_subdivisions[1]*tiling[1] - 1 and 0 < j < domain_subdivisions[0]*tiling[0] - 1:
        draw_hatched_boundary(x, y, X, Y, no='right')
        draw_hatched_boundary(x - boxsize, y, X, Y, no='left')
    elif j == 0 and i == 0:
        draw_hatched_boundary(x, y, X, Y, no=('left', 'bottom'))
        draw_hatched_boundary(x + boxsize, y + boxsize, X, Y, no=('right', 'top'))
        draw_hatched_boundary(x + boxsize, y, X, Y, no=('right', 'bottom'))
        draw_hatched_boundary(x, y + boxsize, X, Y, no=('left', 'top'))
    elif j == 0 and i == domain_subdivisions[1]*tiling[1] - 1:
        draw_hatched_boundary(x, y, X, Y, no=('right', 'bottom'))
        draw_hatched_boundary(x - boxsize, y, X, Y, no=('left', 'bottom'))
        draw_hatched_boundary(x, y + boxsize, X, Y, no=('right', 'top'))
        draw_hatched_boundary(x - boxsize, y + boxsize, X, Y, no=('left', 'top'))
    elif i == domain_subdivisions[1]*tiling[1] - 1 and j == domain_subdivisions[0]*tiling[0] - 1:
        draw_hatched_boundary(x, y, X, Y, no=('right', 'top'))
        draw_hatched_boundary(x - boxsize, y, X, Y, no=('left', 'top'))
        draw_hatched_boundary(x, y - boxsize, X, Y, no=('right', 'bottom'))
        draw_hatched_boundary(x - boxsize, y - boxsize, X, Y, no=('left', 'bottom'))
    elif i == 0 and j == domain_subdivisions[0]*tiling[0] - 1:
        draw_hatched_boundary(x, y, X, Y, no=('left', 'top'))
        draw_hatched_boundary(x + boxsize, y, X, Y, no=('right', 'top'))
        draw_hatched_boundary(x, y - boxsize, X, Y, no=('left', 'bottom'))
        draw_hatched_boundary(x + boxsize, y - boxsize, X, Y, no=('right', 'bottom'))
    # Highlight 3x3 tile block
    i = int(x/tile_size_x)
    j = int(y/tile_size_y)
    for ii in (i - 1, i, i + 1):
        for jj in (j - 1, j, j + 1):
            X = np.array([
                tile_size_x*ii,
                tile_size_x*(ii + 1),
                tile_size_x*(ii + 1),
                tile_size_x*ii,
                tile_size_x*ii,
            ])
            Y = np.array([
                tile_size_y*jj,
                tile_size_y*jj,
                tile_size_y*(jj + 1),
                tile_size_y*(jj + 1),
                tile_size_y*jj,
            ])
            if any(X < 0):
                X += boxsize
            elif any(X >= boxsize*1.001):
                X -= boxsize
            if any(Y < 0):
                Y += boxsize
            elif any(Y >= boxsize*1.001):
                Y -= boxsize
            plt.fill(X, Y, color=color, edgecolor='none', zorder=-91, alpha=0.18)

def connect_particles(POS, color1, color2, lw=1.75):
    points = np.array([
        np.linspace(POS[0][0], POS[1][0], 100),
        np.linspace(POS[0][1], POS[1][1], 100),
    ]).T.reshape(-1, 1, 2)
    gradient = np.linspace(0, 1, points.shape[0])
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    for i in range(segments.shape[0] - 1):
        segments[i, 1, :] += 0.15*(segments[i + 1, 1, :] - segments[i, 1, :])
    lc = LineCollection(
        segments,
        cmap=LinearSegmentedColormap.from_list('mycmap', [(0.0, color2), (1.0, color1)]),
        norm=plt.Normalize(gradient.min(), gradient.max()),
    )
    lc.set_array(gradient)
    lc.set_linewidth(lw)
    lc.set_zorder(303)
    plt.gca().add_collection(lc)



###################
# Place particles #
###################
n_hatches = 6
# Lonely particle
POS_A = [(97*Mpc, 403.7*Mpc), ]
place_particle(*POS_A[0], 'C8', None)  #'\\'*n_hatches)
# Paired particles interacting
POS_B = [(365*Mpc, 299*Mpc), (332*Mpc, 328*Mpc)]
place_particle(*POS_B[0], 'C3', '/'*n_hatches)
place_particle(*POS_B[1], 'C0', '\\'*n_hatches)
connect_particles(POS_B, 'C3', 'C0', lw=1.75)
# Non-interacting particles paired due to insufficient subtiling
POS_C = [(67*Mpc, 151*Mpc), (160*Mpc, 79*Mpc)]
place_particle(*POS_C[0], 'C6', '\\'*n_hatches)
place_particle(*POS_C[1], 'C9', '/'*n_hatches)
plt.plot(*zip(*POS_C), '-', color=[0.35]*3, zorder=303, lw=1.4)
# Two particles not paired due to subtiling
POS_D = [(355*Mpc, 30*Mpc), (445*Mpc, 72.5*Mpc)]
place_particle(*POS_D[0], 'C2', '\\'*n_hatches)
for y in (0, boxsize): 
    plt.plot(  # fixup
        [POS_D[0][0] - shortrange_range*1.5, POS_D[0][0] + shortrange_range*1.5],
        [y]*2,
        'k-', lw=lw_box, zorder=301,
    )
place_particle(*POS_D[1], 'C4', '/'*n_hatches)
plt.plot(*zip(*POS_D), ls=(1.43, np.array((3.97, 1.7))*0.96),
    color=[0.35]*3, zorder=303, lw=1.4)
# Lonely particle
POS_E = [(boxsize/gridsize*27.10, boxsize/gridsize*17.24), ]
place_particle(*POS_E[0], 'C7', None, place_dot=False)

# NGP, CIC, TSC, PCS around particle E
theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
costheta = np.cos(theta)
sintheta = np.sin(theta)
L_varphi = boxsize/gridsize
q_offset = 30*np.pi/180
def mass_assign(pos_x, pos_y, r_scale, weight_function, color, offset=0):
    for i in range(gridsize):
        x = (i + 0.5)*boxsize/gridsize
        w_x = weight_function(abs(pos_x - x)/L_varphi)
        if w_x == 0:
            continue
        for j in range(gridsize):
            y = (j + 0.5)*boxsize/gridsize        
            w_y = weight_function(abs(pos_y - y)/L_varphi)
            if w_y == 0:
                continue
            w = w_x*w_y
            plt.fill(
                x + np.sqrt(w)*r_scale*costheta + offset*np.cos(q_offset),
                y + np.sqrt(w)*r_scale*sintheta + offset*np.sin(q_offset),
                color,
                ec='none', zorder=np.inf)
def W_NGP(x):
    if x < 0.5:
        return 1
    return 0
def W_CIC(x):
    if x < 1:
        return 1 - x
    return 0
def W_TSC(x):
    if x < 0.5:
        return 0.75 - x**2
    elif x < 1.5:
        return 0.125*(2*x - 3)**2
    return 0
def W_PCS(x):
    if x < 1:
        return 1/6*(3*x**3 - 6*x**2 + 4)
    elif x < 2:
        return 1/6*(2 - x)**3
    return 0
offset = 2.2*Mpc
#mass_assign(POS_E[0][0], POS_E[0][1], eps_soft, W_NGP, 'k', 0*offset)
#mass_assign(POS_E[0][0], POS_E[0][1], eps_soft, W_CIC, 'b', 1*offset)
#mass_assign(POS_E[0][0], POS_E[0][1], eps_soft, W_TSC, 'g', 2*offset)
#mass_assign(POS_E[0][0], POS_E[0][1], eps_soft, W_PCS, 'r', 3*offset)
mass_assign(POS_E[0][0], POS_E[0][1], eps_soft, W_PCS, 'C7')

# Fill in ghost layers around one domain
def ghostly(x, y, ax=None):
    if ax is None:
        ax = ply.gca()
    poly, = ax.fill(x, y, facecolor='none', edgecolor='none')
    shape = (20*gridsize, )*2
    img_data = np.random.random(shape)
    scale = 30.0
    octaves = 8
    persistence = 0.65
    lacunarity = 2.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            img_data[i][j] = noise.pnoise2(
                i/scale, 
                j/scale, 
                octaves=octaves, 
                persistence=persistence, 
                lacunarity=lacunarity, 
                repeatx=1024, 
                repeaty=1024, 
                base=0,
            )
    img_data = np.roll(img_data, 5*20, 0)
    img_data = np.roll(img_data, 5*20, 1)
    im = ax.imshow(
        img_data, aspect='auto', origin='lower',
        cmap=plt.cm.binary,
        extent=[1e-2, boxsize - 1e-2, 1e-2, boxsize - 1e-2],
        vmin=-0.25, vmax=1.5,
        zorder=-np.inf,
    )
    im.set_clip_path(poly)
ghostly(
    boxsize/gridsize*np.array([16, 38, 38, 36, 36, 18, 18, 16, 16,   16, 16, 38, 38, 16]),
    boxsize/gridsize*np.array([29, 29,  0,  0, 27, 27,  0,  0, 29,   52, 54, 54, 52, 52]),
    plt.gca(),
)

## Labels
"""
# Box
space = r'\,'*104
fontsize = 12
plt.text(1.023*boxsize, 0.5*boxsize, rf'$\overbrace{{{space}}}^{{\,}}$',
    fontsize=14, ha='center', va='center', rotation=-90)
plt.text(1.054*boxsize, 0.5*boxsize, r'$L_{\text{box}}$',
    fontsize=fontsize, ha='left', va='center')
# Domain
space = r'\,'*34
fontsize = 10
plt.text(1/6*boxsize, 1.023*boxsize, rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=14, ha='center', va='center')
plt.text(1/6*boxsize, 1.05*boxsize, r'$L_{\text{dom}}^x$',
    fontsize=fontsize, ha='center', va='bottom')
# Tile
space = r'\,'*11
fontsize = 8
plt.text(3.5/9*boxsize, 1.023*boxsize, rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=14, ha='center', va='center', color='C5')
plt.text(3.5/9*boxsize, 1.05*boxsize, r'$L_{\text{til}}^x$',
    fontsize=fontsize, ha='center', va='bottom', color='C5')
# Subtile
fontsize = 7
plt.text(8.5/18*boxsize, 1.0264*boxsize, r'$\{$',
    fontsize=13, ha='center', va='center', rotation=-90, color='C1')
plt.text(8.5/18*boxsize, 1.047*boxsize, r'$L_{\text{stil}}^{p,x}$',
    fontsize=fontsize, ha='center', va='bottom', color='C1')
# PM grid cell
plt.text(29.5/54*boxsize, 1.021*boxsize, r'$\{$',
    fontsize=5, ha='center', va='center', rotation=-90, color=[0.89*0.7]*3)
plt.text(29.5/54*boxsize, 1.031*boxsize, r'$L_{\varphi}$',
    fontsize=7, ha='center', va='bottom', color=[0.89*0.7]*3)
"""

"""
## New labels, all inside the box
# Box
space = r'\,'*984
plt.text(1/2*boxsize, 1/2*boxsize + 8.0*Mpc,
    rf'$\aoverbrace[L9999999992U9999999999999991R]{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(1/2*boxsize, 1/2*boxsize + 8.0*Mpc, # fixup
    rf'$\aoverbrace[L203R]{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(0.3668*boxsize, 0.55925*boxsize, r'$L_{\text{box}}^{\phantom{x}}$',
    fontsize=7, ha='left', va='center', color='k', zorder=np.inf)
# Domain
space = r'\,'*489
plt.text(1/3*boxsize + 8.2*Mpc, 2/3*boxsize + 42.5*Mpc,
    rf'$\aoverbrace[L999U992R]{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(1/3*boxsize + 8.2*Mpc, 2/3*boxsize + 42.5*Mpc,  # fixup
    rf'$\aoverbrace[L302R]{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(0.3668*boxsize, 11.5/16*boxsize, r'$L_{\text{dom}}^{\smash{y}}$',
    fontsize=7, ha='left', va='center', color='k', zorder=np.inf)
# Tile
space = r'\,'*11
plt.text(3.5/9*boxsize + 0.15*Mpc, 7/8*boxsize + 7.8*Mpc, rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=13.9, ha='center', va='center', color='k', rotation=0, zorder=np.inf)
plt.text(0.3668*boxsize, 0.920*boxsize + 0.1*Mpc, r'$L_{\text{tile}}^{\smash{x}}$',
    fontsize=7, ha='left', va='bottom', color='k', zorder=np.inf)
# Subtile
plt.text(8.5/18*boxsize + 0.1*Mpc, 0.949*boxsize, r'$\{$',
    fontsize=12.9, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(8.5/18*boxsize + 2.9*Mpc, 0.9625*boxsize,
    r'$L_{\text{sub}}^{\smash{p,x}}$',
    fontsize=7, ha='center', va='bottom', color='k', zorder=np.inf)
# PM grid cell
plt.text(28.5/54*boxsize, 0.895*boxsize, r'$\{$',
    fontsize=4.5, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(28.5/54*boxsize, 0.9035*boxsize, r'$L_{\varphi}$',
    fontsize=7, ha='center', va='bottom', color='k', zorder=np.inf)
"""
## New labels again, to the left and right of the box
# Box
space = r'\,'*989
plt.text(1.027*boxsize, 1/2*boxsize,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(1.057*boxsize, 0.5*boxsize, r'$L_{\text{box}}^{\phantom{x}}$',
    fontsize=7, ha='left', va='center', color='k', zorder=np.inf)
# Domain
space = r'\,'*494
plt.text(-0.027*boxsize, 0.75*boxsize,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=90, color='k', zorder=np.inf)
plt.text(-0.057*boxsize, 0.75*boxsize, r'$L_{\text{dom}}^{\smash{y}}$',
    fontsize=7, ha='right', va='center', color='k', zorder=np.inf)
# Tile
space = r'\,'*123
plt.text(-0.027*boxsize, 5/16*boxsize, rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', color='k', rotation=90, zorder=np.inf)
plt.text(-0.057*boxsize, 5/16*boxsize, r'$L_{\text{tile}}^{\smash{y}}$',
    fontsize=7, ha='right', va='center', color='k', zorder=np.inf)
# Subtile
plt.text(-0.027*boxsize - 0.0007*boxsize, 2.5/16*boxsize - 0.00017*boxsize, r'$\{$',
    fontsize=15.6, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(-0.057*boxsize, 2.5/16*boxsize, r'$L_{\text{sub}}^{\smash{p,y}}$',
    fontsize=7, ha='right', va='center', color='k', zorder=np.inf)
# PM grid cell
plt.text(-0.027*boxsize + 0.0096*boxsize, 1.5/54*boxsize - 0.00018*boxsize, r'$\{$',
    fontsize=4.65, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(-0.057*boxsize + 0.026*boxsize, 1.5/54*boxsize, r'$L_{\varphi}$',
    fontsize=7, ha='right', va='center', color='k', zorder=np.inf)

# Short-range force range (cutoff)
q = 24*np.pi/180
x = [POS_A[0][0], POS_A[0][0] + shortrange_range*np.cos(q)]
y = [POS_A[0][1], POS_A[0][1] + shortrange_range*np.sin(q)]
space = r'\,'*14
plt.text(np.mean(x) - 0.002*boxsize, np.mean(y) + 0.0067*boxsize,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=10.5, ha='center', va='center', rotation=q*180/np.pi, zorder=np.inf, color='k')
plt.text(np.mean(x) - 0.0188*boxsize, np.mean(y) + 0.038*boxsize,
    r'$x_{\text{r}}$',
    fontsize=7, ha='center', va='center', rotation=q*180/np.pi, zorder=np.inf, color='k')

# Softening length
q = 125
offset_r = 12.9*Mpc
plt.text(
    POS_A[0][0] + offset_r*np.cos(q*np.pi/180),
    POS_A[0][1] + offset_r*np.sin(q*np.pi/180),
    r'$\{$',
    fontsize=7.4, ha='center', va='center', rotation=(q-180), color='k', zorder=np.inf)
offset_r += 10.7*Mpc
plt.text(
    POS_A[0][0] + offset_r*np.cos(q*np.pi/180),
    POS_A[0][1] + offset_r*np.sin(q*np.pi/180),
    r'$2\epsilon$',
    fontsize=7, ha='center', va='center', color='k', rotation=(q - 90), zorder=np.inf)

# Single-particle potential
def get_r3inv_softened(r2):
    h = 2.8*eps_soft
    r = np.sqrt(r2)
    if r >= h:
        return 1/(r2*r)
    u = r/h
    if u < 0.5:
        return 32/h**3*(1./3. + u**2*(-6./5. + u))
    return 32/(3*r**3)*(u**3*(2 + u*(-9./2. + u*(18./5. - u))) - 3./480.)
r = np.linspace(-1, +1, 5000)*shortrange_range*1.5
r3_inv_softened = np.array([get_r3inv_softened(ri**2) for ri in r])
x = np.abs(r)/shortrange_scale
F = -r/np.abs(r)**3
F_softened = -r*r3_inv_softened
F_shortrange = -r/np.abs(r)**3*(x/np.sqrt(np.pi)*np.exp(-x**2/4) + erfc(x/2))
F_shortrange_softened = -r*r3_inv_softened*(x/np.sqrt(np.pi)*np.exp(-x**2/4) + erfc(x/2))
pot_fac = 8.20e+2
pot = -scipy.integrate.cumtrapz(F, r, initial=0)*pot_fac
pot_cut = -0.5*boxsize
pot[pot < pot_cut] = np.nan
pot_softened = -scipy.integrate.cumtrapz(F_softened, r, initial=0)*pot_fac
pot_shortrange = -scipy.integrate.cumtrapz(F_shortrange, r, initial=0)*pot_fac
pot_shortrange[pot_shortrange < pot_cut] = np.nan
pot_shortrange_softened = -scipy.integrate.cumtrapz(F_shortrange_softened, r, initial=0)*pot_fac
offset_x = POS_A[0][0]
offset_y = POS_A[0][1]

fac = 1.24
x_left  = offset_x - shortrange_range*fac
x_right = offset_x + shortrange_range*fac
step = 3.30*Mpc
n_dots = int(round((x_right - x_left)/step))
x = np.linspace(x_left, x_right, n_dots)
y = [offset_y]*len(x)
alpha = 1 - np.exp(-(x - offset_x)**2/(2*shortrange_range**2)*7.8)*1.45
alpha[alpha > 1] = 1
alpha[alpha < 0] = 0
for i in range(len(x)):
    plt.plot(x[i], y[i], '.k', alpha=alpha[i], markeredgecolor='none', ms=1.55, zorder=200.5)

mask = (-shortrange_range <= r) & (r <= shortrange_range)
plt.plot(
    offset_x + r[mask],
    offset_y + pot_shortrange_softened[mask],
    'k-', lw=0.8, zorder=500,
)
plt.plot(
    offset_x + r[mask],
    offset_y + pot_softened[mask],
    'k-', lw=0.6, zorder=500,
)
mask_left  = (r < 0) & (r > -shortrange_range) & ~np.isnan(pot)
mask_right = (r > 0) & (r < shortrange_range) & ~np.isnan(pot)
for j, mask in enumerate((mask_left, mask_right)):
    r_masked = r[mask] + offset_x
    pot_masked = pot[mask] + offset_y
    if j == 1:
        r_masked = r_masked[::-1]
        pot_masked = pot_masked[::-1]
    for i in range(1, len(r_masked)):
        x = [r_masked[i - 1], r_masked[i]]
        y = [pot_masked[i - 1], pot_masked[i]]
        y_min = 8.55/16*boxsize  #0.875
        alpha_min = 0.0
        y_max = y_min + 3/16*boxsize
        alpha_max = 0.8
        if y[0] > y_max:
            alpha = alpha_max
        else:
            a = (alpha_max - alpha_min)/(y_max - y_min)
            b = alpha_max - a*y_max
            alpha = a*y[0] + b
            if alpha > 1:
                alpha = 1
        if alpha < 0:
            break
        plt.plot(x, y, color='k', alpha=alpha, lw=0.6, zorder=499)
plt.text(offset_x, offset_y - 23.4*Mpc, r'$\{$',
    fontsize=11.0, ha='center', va='center', rotation=-90, color='k', zorder=np.inf)
plt.text(offset_x, offset_y - 18.4*Mpc, r'$2x_{\text{s}}$',
    fontsize=7, ha='center', va='bottom', color='k', zorder=np.inf)

# Save
plt.axis('square')
plt.axis('off')
plt.savefig('../figure/.geometry.pdf', dpi=350)
os.system('cd ../figure && pdfcrop --margins 0.5 .geometry.pdf geometry.pdf >/dev/null && rm -f .geometry.pdf')

