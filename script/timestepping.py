import collections, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.interpolate



fontsize = 11/1.4
latex_preamble = r'''
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{mathtools}
    \usepackage{bm}
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



fig, ax = plt.subplots(1, 1, figsize=(8.44, 3.9))

def get_aspect(ax=None):
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()
    return aspect

def draw_arrow(x, y, dir, color='k', rot=None, sync=False, zorder=None):
    if zorder is None:
        zorder = -15
    text = (r'$\bm{\uparrow}$' if dir == 'up' else r'$\bm{\downarrow}$')
    va = ('top' if dir == 'up' else 'bottom')
    fontsize = 19
    if sync:
        fontsize = 14.9
    if rot is not None:
        v = [np.cos(rot*np.pi/180), np.sin(rot*np.pi/180)/get_aspect()]
        t = -0.034 #-0.053
        dy = -0.001
        plt.text(
            x + t*v[0], y + t*v[1] + dy,
            r'$\bm{\rightarrow}$',
            va='center', ha='center', fontsize=fontsize,
            zorder=zorder, color=color, rotation=rot,
        )
        # Hide stalk
        if not sync:
            for dt in (-0.0056, ):
                plt.text(
                    x + (t + dt)*v[0], y + (t + dt)*v[1] + dy,
                    r'$\bm{-}$',
                    va='center', ha='center', fontsize=22,
                    zorder=zorder+1, color='w', rotation=rot,
                )
            for dt in (-0.036, ):
                plt.text(
                    x + (t + dt)*v[0], y + (t + dt)*v[1] + dy,
                    r'$\bm{-}$',
                    va='center', ha='center', fontsize=36,
                    zorder=zorder+1, color='w', rotation=rot,
                )
        return
    # Not rotated
    plt.text(
        x, y, text,
        va=va, ha='center', fontsize=fontsize,
        zorder=zorder, color=color,
    )
    # Hide stalk
    if not sync:
        dx = 0.010
        dy = 0.192
        dY = (-0.145 if dir == 'up' else +0.145)
        plt.fill(
            [x - 0.5*dx, x + 0.5*dx, x + 0.5*dx, x - 0.5*dx, x - 0.5*dx],
            np.array([y + 0.5*dy, y + 0.5*dy, y - 0.5*dy, y - 0.5*dy, y + 0.5*dy]) + dY,
            'w', ec='none', zorder=zorder+1,
        )
        dY += 0.1*dY
        dx *= 1.3
        plt.fill(
            [x - 0.5*dx, x + 0.5*dx, x + 0.5*dx, x - 0.5*dx, x - 0.5*dx],
            np.array([y + 0.5*dy, y + 0.5*dy, y - 0.5*dy, y - 0.5*dy, y + 0.5*dy]) + dY,
            'w', ec='none', zorder=zorder+1,
        )

theta = np.linspace(np.pi, 0, 201)
def step(bgn, end, offset_y, dir, color, colors=None, jump_up=False, jump_down=False):
    global y_jump_up_last, y_jump_down_last
    arrow_offset = 0.04
    jump_up_height = 0.10 #0.0925 #0.135
    if offset_y == offset_y0:
        jump_down_height = 0.79 - 0.05
    else:
        jump_down_height = 0.614 + 0.018 - 0.05
    if offset_y == offset_y2:
        jump_up_height += 0.013 #0.008
    x = bgn + ((end - bgn)/2)*(1 + np.cos(theta))
    if dir == 'up':
        y = (height/2)*np.sin(theta)
    elif dir == 'down':
        y = -(height/2)*np.sin(theta)
    else:
        print(f'Unrecognized dir="{dir}"', file=sys.stderr, flush=True)
        sys.exit(1)
    y += offset_y
    if colors:
        color0, color1 = colors
        color0 = np.asarray(matplotlib.colors.ColorConverter().to_rgb(color0), dtype=float)
        color1 = np.asarray(matplotlib.colors.ColorConverter().to_rgb(color1), dtype=float)
        mid = (x.size - 1)/2
        for i in range(x.size - 1):
            w = (1 + erf(1.8*(i - mid)/mid))/2
            color = (1 - w)*color0 + w*color1
            plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], '-', color=color, lw=1.2)
            # Arrow
            if i == int((x.size - 1)*0.30):
                dy = (y[i+1] - y[i-1])/2*get_aspect()
                dx = (x[i+1] - x[i-1])/2
                draw_arrow(x[i], y[i], 'up', color, rot=180/np.pi*np.arctan2(dy, dx))
        el_skip = 16
        if jump_up:
            if jump_up is True:
                y_jump = np.array(
                      list(y[:len(y)//2])
                    + list(offset_y + np.linspace(
                          height/2,
                          height/2 + jump_up_height,
                          len(y) - len(y)//2,
                      ))
                )
                X = bgn + (end - bgn)/2
                x_jump = np.array(list(x[:len(x)//2]) + [X]*(len(x) - len(x)//2))
                mid = (y_jump.size - 1)/2
                random_fac = 1.22  # because I can't do the math, apparently
                mid *= random_fac
                for i in range(len(y)//2 + el_skip, y_jump.size - 1):
                    w = (1 + erf(1.95*(i - mid)/mid))/2
                    color = (1 - w)*color0 + w*color1
                    plt.plot([x_jump[i], x_jump[i+1]], [y_jump[i], y_jump[i + 1]],
                        '-', color=color, lw=1.2)
                # Arrow
                draw_arrow(x_jump[i+1], y_jump[i+1] + arrow_offset, 'up', color1)
            else:
                X1 = bgn + (jump_up - bgn)/2
                index1 = np.argmin((X1 - x)**2)
                x_jump = np.array([X1]*(len(x)//2))
                y_jump = np.linspace(
                    offset_y + height/2 + 1e-3,
                    y_jump_up_last[-1],  #offset_y + height/2 + jump_up_height,
                    x_jump.size,
                 )
                mid = (y_jump.size - 1)/2
                random_fac = 1.22  # because I can't do the math, apparently
                for i in range(y_jump.size - 1):
                    w = (1 + erf(1.95*(i - mid)/mid))/2
                    color = (1 - w)*(color0/(1 + random_fac*index1/len(x_jump))) + w*color1
                    plt.plot([x_jump[i], x_jump[i+1]], [y_jump[i], y_jump[i + 1]],
                        '-', color=color, lw=1.2)
                # Arrow
                draw_arrow(x_jump[i+1], y_jump[i+1] + arrow_offset, 'up', color1)
            y_jump_up_last = y_jump
        if jump_down:
            if jump_down is True:
                X = bgn + (end - bgn)*3/4
                x_jump = np.array(list(x[:3*len(x)//4]) + [X]*(len(x) - 3*len(x)//4))
                Y = np.interp(X, x, y)
                y_jump = np.array(
                      list(y[:3*len(y)//4])
                    + list(np.linspace(
                          Y - 2e-3,
                          Y - jump_down_height,
                          len(y) - 3*len(y)//4,
                      ))
                )
                mid = (y_jump.size - 1)/2
                for i in range(3*len(y)//4, y_jump.size - 1):
                    w = (1 + erf(1.4*(i - mid)/mid))/2
                    color = (1 - w)*color0 + w*color1
                    plt.plot([x_jump[i], x_jump[i+1]], [y_jump[i], y_jump[i + 1]],
                        '-', color=color, lw=1.2)
                # Arrow
                draw_arrow(x_jump[i+1], y_jump[i+1] - arrow_offset, 'down', color1)
            else:
                X1 = bgn + 3*(jump_down - bgn)/4
                Y = np.interp(X1, x, y)
                index1 = np.argmin((X1 - x)**2)
                x_jump = np.array([X1]*(1*len(x)//2))
                y_jump = np.linspace(Y - 2e-3, y_jump_down_last[-1], len(x_jump))
                mid = (y_jump.size - 1)/2
                random_fac = 3.70  # because I can't do the math, apparently
                for i in range(y_jump.size - 1):
                    w = (1 + erf(1.4*(i - mid)/mid))/2
                    color = (1 - w)*(color0/(1 + random_fac*index1/len(x_jump))) + w*color1
                    plt.plot([x_jump[i], x_jump[i+1]], [y_jump[i], y_jump[i + 1]],
                        '-', color=color, lw=1.2)
                # Arrow
                draw_arrow(x_jump[i+1], y_jump[i+1] - arrow_offset, 'down', color1)
            y_jump_down_last = y_jump
    else:
        plt.plot(x, y, '-', color=color, lw=1.2)
        # Arrow
        i = int((x.size - 1)*0.33)
        dy = (y[i+1] - y[i])*get_aspect()
        dx = (x[i+1] - x[i])
        draw_arrow(x[i], y[i], 'down', color, rot=180/np.pi*np.arctan2(dy, dx))
y_jump_up_last = None
y_jump_down_last = None

# Specs
height = 0.615  #0.68
rung_offset = 0.75
rung0_final_step = 0.5 #0.21 #0.457
offset_y0 = 0
offset_y1 = -1.102*rung_offset
offset_y2 = -2*rung_offset
offset_ydrift = -2.73*rung_offset
end_sync = 1/2 + 1 + 1 + rung0_final_step
particle_scatter_size = 98
particle_vert_offset = 0.0135*np.sqrt(particle_scatter_size)
dy_vert = 0.085 #0.079
dy_vert_fac = 1.2
dx_rung0 = 0.0567 # 0.0507
dx_rung1 = 0.033 #0.0295
colors = ['C0', 'C1', 'C2', 'C3']

# Curve through blue points
lw_fat = 14.5
alpha_fat = 0.154
def draw_fat_blue_curve(x_offset):
    dX_up = 0.017 #-0.015 #0.036
    dX_down = -0.006
    dY_up = 0.1 #0.22
    dY_down = 0.1
    X = [
        1.0*dX_down + 1 - 0.015,
        1 + 0.4*dX_down,
        #
        1,
        1 + 1/8,
        0.2*(2*(1 + 1/4) + 3*(1 + 1/4 - dx_rung1)),
        0.2*(2*(1 + 1/2) + 3*(1 + 1/2 - dx_rung0)),
        #
        #(1 + 1/2),
        #(1 + 1/2),
        dX_up + (1 + 1/2),
    ]
    X = np.array(X) + x_offset
    Y = [
        -1.0*dY_down + offset_ydrift + 0.0,
        -0.4*dY_down + offset_ydrift + 0.03,
        #
        0.05 + 0.2*(2*(offset_ydrift) + 3*(offset_ydrift + dy_vert_fac*dy_vert)) + 0.03,
        0.2*(2*(offset_y2) + 3*(offset_y2 - dy_vert_fac*dy_vert)) + 0.03,
        0.2*(2*(offset_y1) + 3*(offset_y1 - dy_vert_fac*dy_vert)),
        0.2*(2*(offset_y0) + 3*(offset_y0 - dy_vert*(1 + dy_vert_fac))),
        #
        #offset_y0,
        #0.4*dY_up + offset_y0,
        1.0*dY_up + offset_y0,
    ]
    tck, u = scipy.interpolate.splprep([X, Y], s=1.58e-3, k=2)
    unew = np.arange(0, 1.01, 0.01)
    out = scipy.interpolate.splev(unew, tck)
    color_C0 = np.asarray(matplotlib.colors.ColorConverter().to_rgb('C0'), dtype=float)
    color_c = np.asarray(matplotlib.colors.ColorConverter().to_rgb('c'), dtype=float)
    w = 0.66
    color = w*color_C0 + (1 - w)*color_c
    plt.plot(out[0], out[1], '-', color=color, lw=lw_fat, alpha=alpha_fat, zorder=-12.9, solid_capstyle='round')
draw_fat_blue_curve(0)
draw_fat_blue_curve(1)
# Black curves
plt.plot([0, 0], [offset_ydrift - 0.1, offset_y0 + 0.1],
    'k', lw=lw_fat, alpha=alpha_fat, zorder=-12.9, solid_capstyle='round')
plt.plot([end_sync, end_sync], [offset_ydrift - 0.1, offset_y0 + 0.1],
    'k', lw=lw_fat, alpha=alpha_fat, zorder=-12.9, solid_capstyle='round')

# Labels
x = -0.085
dy = 0.123
fontsize = 11
plt.text(x, offset_y0 - dy, 'rung 0',
    va='bottom', ha='right', fontsize=fontsize, rotation=90)
plt.text(x - 0.067, offset_y0 - dy, 'long-range,',
    va='bottom', ha='right', fontsize=fontsize, rotation=90)
plt.text(x, offset_y1 - dy, 'rung 1',
    va='bottom', ha='right', fontsize=fontsize, rotation=90)
plt.text(x, offset_y2 - dy, 'rung 2',
    va='bottom', ha='right', fontsize=fontsize, rotation=90)
plt.text(x, offset_ydrift + dy, 'drift',
    va='top', ha='right', fontsize=fontsize, rotation=90)
# Delta t
y = 0.529
space = r'\,'*736
plt.text(0.5, y,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(0.5, y + 0.140, r'initial, $\Delta t$',
    fontsize=fontsize, ha='center', va='center', color='k', zorder=np.inf)
plt.text(1.5, y,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(1.5, y + 0.140, r'repeatable, $\Delta t$',
    fontsize=fontsize, ha='center', va='center', color='k', zorder=np.inf)
space = r'\,'*int(round(len(space)/2*(end_sync - 2)/1) - 1)
plt.text(0.5*(2 + end_sync), y,
    rf'$\overbrace{{{space}}}^{{}}$',
    fontsize=1, ha='center', va='center', rotation=0, color='k', zorder=np.inf)
plt.text(0.5*(2 + end_sync), y + 0.140, r'synchronisation, $\leq\Delta t$',
    fontsize=fontsize, ha='center', va='center', color='k', zorder=np.inf)

# Time step
y = -2.47
plt.text(0, y, r'$t_0$', fontsize=fontsize, ha='center', va='top')
plt.text(1, y, r'$t_1$', fontsize=fontsize, ha='center', va='top')
plt.text(2, y, r'$t_2$', fontsize=fontsize, ha='center', va='top')
plt.text(end_sync, y, r'$t_{\text{sync}}$', fontsize=fontsize, ha='center', va='top')


# For testing for ellipticity
"""
THETA = np.linspace(0, 2*np.pi, 200)
end = 0
for i in range(int(((1/2 + 1 + 1 + rung0_final_step) - 0)//(1/4))):
    bgn, end = end, end + 1/4  # full step
    if i == 3:
        step(bgn, end, offset_ydrift, 'down', 'k')
        R = 3.5
        x = bgn + R*((end - bgn)/2)*(1 + np.cos(THETA))
        y = -1 -R*(height/2)*np.sin(THETA)
        plt.plot(x, y, 'r-', lw=0.5, zorder=np.inf)
        break
"""

# Set axis
def set_axis():
    xlim = -0.06, end_sync + 0.06
    ylim = -2.8, 0.8 # -2.4, 0.55
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis('off')
    plt.tight_layout()
    plt.xlim(xlim)
    plt.ylim(ylim)
set_axis()

# Rung 0
bgn = 0
end = bgn + 1/2
step(bgn, end, offset_y0, 'up', 'k', ('k', colors[0]))  # init
for i in range(2):
    bgn, end = end, end + 1  # full step
    step(bgn, end, offset_y0, 'up', 'k', (colors[3], colors[0]),
        jump_down=True)
bgn, end = end, end + rung0_final_step  # sync step
step(bgn, end, offset_y0, 'up', 'k', (colors[3], 'k'))

# Rung 1
bgn = 0
end = bgn + 1/4
step(bgn, end, offset_y1, 'up', 'k', ('k', colors[0]))  # init
for i in range(int(((1/2 + 1 + 1 + rung0_final_step) - 1/4)//(1/2))):
    bgn, end = end, end + 1/2  # full step
    step(bgn, end, offset_y1, 'up', 'k',
        (colors[3], colors[0]) if i%2 else (colors[1], colors[2]),
        jump_up=(not i%2),
        jump_down=True,
    )
bgn, end = end, end_sync  # sync step
#step(bgn, end, offset_y1, 'up', 'k', (colors[1], colors[2]),
#    jump_up=(bgn + 1/2), jump_down=(bgn + 1/2))
step(bgn, end, offset_y1, 'up', 'k', (colors[3], 'k'))

# Rung 2
bgn = 0
end = bgn + 1/8
step(bgn, end, offset_y2, 'up', 'k', ('k', colors[0]))  # init
for i in range(int(((1/2 + 1 + 1 + rung0_final_step) - 1/8)//(1/4))):
    bgn, end = end, end + 1/4  # full step
    step(bgn, end, offset_y2, 'up', 'k', (colors[i%4], colors[(i+1)%4]),
        jump_up=(not i%2))
bgn, end = end, end_sync  # sync step
step(bgn, end, offset_y2, 'up', 'k', (colors[3], 'k'))

# Drifts
end = 0
for i in range(int(((1/2 + 1 + 1 + rung0_final_step) - 0)//(1/4))):
    bgn, end = end, end + 1/4  # full step
    step(bgn, end, offset_ydrift, 'down', 'k')
#bgn, end = end, end_sync  # sync step
#step(bgn, end, offset_ydrift, 'down', 'k')

# Vertical lines
color_vert = [0.47]*3  # 'grey'
lw_vert = 1.0
# Sync lines
for x in (0, end_sync):
    plt.plot([x]*2, [-2.33 - 0.102 + 0.02, 0.34 + 0.102], '-', color=color_vert, lw=lw_vert, zorder=-16)
# Fixups due to hiding of arrow stalks
plt.plot([0]*2, [0.1, 0.3], '-', color=color_vert, lw=lw_vert, zorder=-13)
plt.plot([0]*2, [-0.8, -0.5], '-', color=color_vert, lw=lw_vert, zorder=-13)
plt.plot([0]*2, [-1.4, -1.26], '-', color=color_vert, lw=lw_vert, zorder=-13)
plt.plot([0]*2, [-2.3, -2.1], '-', color=color_vert, lw=lw_vert, zorder=-13)
# Full time step indicaters
for i in range(1, 3):
    plt.plot([i]*2, [-2.33 - 0.102 + 0.02, 0.34 + 0.102], '--', color=color_vert,
        lw=lw_vert, zorder=-13)
# Horizontal separator between kicks and drifts
dots = np.linspace(0, end_sync, 108)[1:-1]
plt.plot(dots, [0.5*(offset_y2 + offset_ydrift)]*len(dots), '.',
    color=color_vert, zorder=-13, ms=2.0, lw=0,  markeredgecolor='none')

# Vertical black arrows
"""
blackarrow_dy = 0.153
#
y1 = offset_ydrift + dy_vert_fac*dy_vert
y2 = offset_y2 - dy_vert_fac*dy_vert
plt.plot([0, 0], [y1, y2], 'k', lw=lw_vert, zorder=-10)
y1 += blackarrow_dy
y2 -= blackarrow_dy
blackarrow_dy_between = y2 - y1
draw_arrow(0, y1, 'up', color='k', sync=True)
draw_arrow(0, y2, 'down', color='k', sync=True)
#
y1 = offset_y2 - dy_vert_fac*dy_vert
y2 = offset_y1 - dy_vert_fac*dy_vert
y3 = 0.5*(y1 + y2) - 0.5*blackarrow_dy_between
y4 = 0.5*(y1 + y2) + 0.5*blackarrow_dy_between
draw_arrow(0, y3, 'up', color='k', sync=True, zorder=-13.9)
draw_arrow(0, y4, 'down', color='k', sync=True)
plt.plot([0, 0], [y1, y2], 'k', lw=lw_vert, zorder=-10)
#
y1 = offset_y1 - dy_vert_fac*dy_vert
y2 = offset_y0 - dy_vert_fac*dy_vert
y3 = 0.5*(y1 + y2) - 0.5*blackarrow_dy_between
y4 = 0.5*(y1 + y2) + 0.5*blackarrow_dy_between
draw_arrow(0, y3, 'up', color='k', sync=True, zorder=-13.9)
draw_arrow(0, y4, 'down', color='k', sync=True)
plt.plot([0, 0], [y1, y2], 'k', lw=lw_vert, zorder=-10)
"""

# Particles
bank = collections.Counter()
#for step in range(1, 4):
#    bank[0, step] = 4 - 1
#for step in range(1, 7):
#    bank[1, step] = 2 - 1
def draw_particle(rung, step, color, hatch=None):
    lw = 0.135*np.sqrt(particle_scatter_size)
    x = 0
    y = 0
    y += particle_vert_offset*bank[rung, step]
    if rung == 0:
        y -= particle_vert_offset*bank[rung, step]
        dx = dx_rung0
        y -= dy_vert_fac*dy_vert
        if bank[rung, step] == 0:
            if 0 < step < 4 and step != 2.5:
                x -= dx
                y -= dy_vert
        elif bank[rung, step] == 1:
            pass
        elif bank[rung, step] == 2:
            y -= 2*dy_vert
        elif bank[rung, step] == 3:
            x += dx
            y -= dy_vert
    elif rung == 1:
        y -= particle_vert_offset*bank[rung, step]
        dx = dx_rung1
        y -= dy_vert_fac*dy_vert
        if bank[rung, step] == 0 and step > 0:
            x -= dx
        elif bank[rung, step] == 1:
            x += dx
    elif rung == 2:
        y -= particle_vert_offset*bank[rung, step]
        y -= dy_vert_fac*dy_vert
    elif rung == 'drift':
        y -= particle_vert_offset*bank[rung, step]
        y += dy_vert_fac*dy_vert

    #bank[rung, step] -= 1
    bank[rung, step] += 1
    ec = 0.90*np.asarray(matplotlib.colors.ColorConverter().to_rgb(color), dtype=float)
    if rung == 0:
        y += offset_y0
    elif rung == 1:
        y += offset_y1
    elif rung == 2:
        y += offset_y2
    elif rung == 'drift':
        y += offset_ydrift
    else:
        print(f'Could not understand rung = {rung}', file=sys.stderr, flush=True)
        sys.exit(1)
    if rung == 'drift':
        x += 1/4*step
    else:
        if step > 0:
            x += 1/2**(rung + 1)
        if step > 1:
            x += 1/2**rung*(step - 1)
    if x > end_sync:
        x = end_sync
    marker = 'o'
    if rung == 'drift':
        marker = 'h'
    plt.scatter(x, y, particle_scatter_size, c='w', marker=marker,
        edgecolors='w', lw=lw, zorder=10)
    alpha = 0.65
    plt.scatter(x, y, particle_scatter_size, c=color, marker=marker,
        alpha=alpha, edgecolors='None', zorder=10)
    if hatch is not None:
        theta_hatch = np.linspace(0, 2*np.pi, 50)
        r_hatch = 0.025
        aspect = get_aspect()
        matplotlib.rcParams['hatch.linewidth'] = 0.93
        for hatch_color, hatch_alpha in [('w', 1), (hatch, alpha)]:
            plt.fill(
                x + r_hatch*np.cos(theta_hatch),
                y + r_hatch/aspect*np.sin(theta_hatch),
                color='none', edgecolor=hatch_color, zorder=10.1, hatch='/'*8,
                fill=False, lw=0, alpha=hatch_alpha,
            )
        # Manual hatch as dotted hatching apparently
        # does not work properly with PDF.
        """
        r_hatch = 0.025
        n_hatch = 6
        for hatch_color, hatch_alpha in [('w', 1), (hatch, alpha)]:
            X = np.linspace(-2.3*r_hatch, +2*r_hatch, 2*n_hatch)
            Y = np.linspace(-2.3*r_hatch/aspect, +2*r_hatch/aspect, 2*n_hatch)
            Y -= 0.015
            X += 0.0025
            for xx in X:
                for j, yy in enumerate(Y):
                    x_offset = 0
                    if j%2:
                        x_offset = 0.5*(X[1] - X[0])
                    xxx = xx + x_offset
                    if xxx**2 + (yy*aspect)**2 > (0.98*r_hatch)**2:
                        continue
                    plt.scatter(x + xxx, y + yy, 0.015*particle_scatter_size,
                        c=hatch_color, edgecolors='r', lw=0, zorder=10.1,
                        alpha=hatch_alpha)
        """
    plt.scatter(x, y, particle_scatter_size, marker=marker,
        facecolors='none', edgecolors=ec, lw=lw, zorder=10.2)



########################
# Particle "positions" #
########################
# At initial point
draw_particle(0,       0, 'k')
draw_particle(1,       0, 'k')
draw_particle(2,       0, 'k')
draw_particle('drift', 0, 'k', hatch=colors[0])
# Init step
draw_particle(0,       1, colors[0])
draw_particle(1,       1, colors[0])
draw_particle(2,       1, colors[0])
draw_particle('drift', 1, colors[1])
# Rung 2 step + drift
draw_particle(0,       1, colors[1])
draw_particle(1,       1, colors[1])
draw_particle(2,       2, colors[1])
draw_particle('drift', 2, colors[2])
# Rung 2+1 step + drift
draw_particle(0,       1, colors[2])
draw_particle(1,       2, colors[2])
draw_particle(2,       3, colors[2])
draw_particle('drift', 3, colors[3])
# Rung 2 step + drift
draw_particle(0,       1, colors[3])
draw_particle(1,       2, colors[3])
draw_particle(2,       4, colors[3])
draw_particle('drift', 4, colors[0])
# Rung 2+1+0 step + drift
draw_particle(0,       2, colors[0])
draw_particle(1,       3, colors[0])
draw_particle(2,       5, colors[0])
draw_particle('drift', 5, colors[1])
# Rung 2 step + drift
draw_particle(0,       2, colors[1])
draw_particle(1,       3, colors[1])
draw_particle(2,       6, colors[1])
draw_particle('drift', 6, colors[2])
# Rung 2+1 step + drift
draw_particle(0,       2, colors[2])
draw_particle(1,       4, colors[2])
draw_particle(2,       7, colors[2])
draw_particle('drift', 7, colors[3])
# Rung 2 step + drift
draw_particle(0,       2, colors[3])
draw_particle(1,       4, colors[3])
draw_particle(2,       8, colors[3])
draw_particle('drift', 8, colors[0])
# Rung 2+1+0 step + drift
draw_particle(0,       3, colors[0])
draw_particle(1,       5, colors[0])
draw_particle(2,       9, colors[0])
draw_particle('drift', 9, colors[1])
# Rung 2 step + drift
draw_particle(0,        3, colors[1])
draw_particle(1,        5, colors[1])
draw_particle(2,       10, colors[1])
draw_particle('drift', 10, colors[2])
# Rung 2+1 step + drift
draw_particle(0,        3, colors[2])
draw_particle(1,        6, colors[2])
draw_particle(2,       11, colors[2])
draw_particle('drift', 11, colors[3])
# Rung 2 step + drift
draw_particle(0,        3, colors[3])
draw_particle(1,        6, colors[3])
draw_particle(2,       12, colors[3])
draw_particle('drift', 12, 'k')
# Rung 2+1+0 step
draw_particle(0,        4, 'k')
draw_particle(1,        7, 'k')
draw_particle(2,       13, 'k')

# Sync point
"""
dx_rung0_bak = dx_rung0; dx_rung0 = 0
draw_particle(0, 2.5, 'k')
dx_rung0 = dx_rung0_bak
dx_rung1_bak = dx_rung1; dx_rung1 = 0
draw_particle(1, 4.5, 'k')
dx_rung1 = dx_rung1_bak
draw_particle(2, 8.5, 'k')
draw_particle('drift', 8, 'k', hatch=colors[0])
"""

# Save figure
set_axis()
plt.savefig('../figure/.timestepping.pdf', dpi=350)
os.system('cd ../figure && pdfcrop --margins 0.5 .timestepping.pdf timestepping.pdf >/dev/null && rm -f .timestepping.pdf')

