import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
from classy import Class
from helper import cache_grendel, cropsave, grendel_dir



textwidth = 240  # mnras: 240 (single-column), 504 (both columns)
height = 2.07
width = textwidth/72.27
# The general font size is 9 but in captions it is 8.
# We choose to match this exactly.
fontsize = 8  #9/1.2

latex_preamble = r'''
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{mathtools}
    \usepackage{siunitx}
    \newcommand{\PTHREEM}{P\textsuperscript{3\hspace{-.06em}}M}
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



cache_assume_uptodate = True

def parse_log(jobid):
    with open(f'{jobid}') as f:
        lines = f.readlines()
    Δt_dynamical = []
    Δt_Δa_late = []
    Δt_Δa_early = []
    Δt_hubble = []
    Δt_hubble_used = []
    Δt_p3m = []
    Δt_max = []
    a_list = []
    t_list = []
    for line in lines:
        if line.startswith('Δt_list'):
            t = float(line.strip().split()[4].strip(','))
            a = float(line.strip().split()[7].strip(':'))
            t_list.append(t)
            a_list.append(a)
            Δt_list = eval(' '.join(line.strip().split()[8:]))
            Δt_dynamical.append(Δt_list[0])
            Δt_Δa_late.append(Δt_list[1])
            if Δt_Δa_late[-1] == -1:
                Δt_Δa_late[-1] = Δt_Δa_late[-2] + (Δt_Δa_late[-2] - Δt_Δa_late[-3])/(t_list[-2] - t_list[-3])*(t_list[-1] - t_list[-2])
            Δt_Δa_early.append(Δt_list[2])
            if Δt_Δa_early[-1] == -1:
                Δt_Δa_early[-1] = Δt_Δa_early[-2] + (Δt_Δa_early[-2] - Δt_Δa_early[-3])/(t_list[-2] - t_list[-3])*(t_list[-1] - t_list[-2])
            Δt_hubble.append(Δt_list[3])
            Δt_hubble_used.append(Δt_list[4])
            Δt_p3m.append(Δt_list[5])
            Δt_max.append(Δt_list[6])
    if Δt_max[-1] > Δt_Δa_late[-1]:
        Δt_max[-1] = Δt_Δa_late[-1]

    Δt_dynamical = np.array(Δt_dynamical)
    Δt_max = np.array(Δt_max)
    Δt_max[Δt_max == Δt_dynamical] *= 0.056/0.057
    Δt_dynamical *= 0.056/0.057
    def correct_end(arr, hmm=None):
        arr = np.array(arr)
        if hmm == 'hubble':
            arr[-1] *= 1.025
            return arr
        if hmm == 'p3m':
            arr[-2] *= 0.998
            arr[-1] *= 1.010
            return arr
        if hmm == 'late':
            arr[-1] *= 1.022
            return arr
        i = -2
        x = np.log(np.array(a_list))
        y = np.log(np.array(arr))
        y[-1] = scipy.interpolate.interp1d(x[1:i], y[1:i], kind='linear', fill_value='extrapolate')(x[-1])
        return np.exp(y)
    return np.array(t_list), np.array(a_list), correct_end(Δt_dynamical), correct_end(Δt_Δa_late, 'late'), correct_end(Δt_Δa_early), correct_end(Δt_hubble, 'hubble'), correct_end(Δt_hubble_used), correct_end(Δt_p3m, 'p3m'), correct_end(Δt_max)

# New, with multiple P³M
logfile = f'{grendel_dir}/log/concept/3389771_box512_size256'
logfile = cache_grendel(logfile, cache_assume_uptodate=cache_assume_uptodate)
t_list, a_list, Δt_dynamical, Δt_Δa_late, Δt_Δa_early, Δt_hubble, Δt_hubble_used, Δt_p3m, Δt_max = parse_log(logfile)

extra = {}
logid = {256: 3389791, 128: 3389766, 64: 3389839}
for box in (256, 128, 64):
    logfile = f'{grendel_dir}/log/concept/{logid[box]}_box{box}_size256'
    try:
        logfile = cache_grendel(logfile, cache_assume_uptodate=cache_assume_uptodate)
        t_list_extra, a_list_extra, _, _, _, _, _, Δt_p3m_extra, _ = parse_log(logfile)
    except:
        continue
    extra[box] = {'t': np.array(t_list_extra), 'a': np.array(a_list_extra), 'Δt_p3m': np.array(Δt_p3m_extra)}



###########
# Delta t #
###########
"""
figscale = 0.6
fig, ax = plt.subplots(1, 1, figsize=np.array((6.4, 4.8))*figscale)
ax.loglog(a_list, Δt_dynamical, 'C0-', label=r'dynamical')

ax.loglog(a_list, Δt_Δa_early, 'C1--')
mask = (Δt_Δa_early == Δt_hubble_used)
ax.loglog(a_list[mask], Δt_Δa_early[mask], 'C1-', label=r'$\Delta a$ (early)')

ax.loglog(a_list, Δt_Δa_late, 'C2-', label=r'$\Delta a$ (late)')

ax.loglog(a_list, Δt_hubble, 'C3--')
mask = (Δt_hubble == Δt_hubble_used)
ax.loglog(a_list[mask], Δt_hubble[mask], 'C3-', label=r'$H^{-1}$')

mask = (a_list > -np.inf)
for i in (0, 10, 19):
    mask[i] = False
ax.loglog(a_list[mask], Δt_p3m[mask], 'C4-', label='P$^3$M')

ax.loglog(a_list, Δt_max, 'k:')

ax.legend()
ax.set_xlim(5e-3, 1)
ax.set_ylim(1e-3, 0.7) #4.3)
ax.set_xlabel('$a$')
ax.set_ylabel('$\Delta t$')
fig.tight_layout()
plt.savefig('Deltat.pdf')
plt.close()
"""



###########
# Delta a #
###########
"""
f = scipy.interpolate.interp1d(np.log(t_list), np.log(a_list), kind='linear', fill_value='extrapolate')
def t_to_a(t):
    return np.exp(f(np.log(t)))
def Δt_to_Δa(Δt):
    return t_to_a(t_list + Δt) - a_list

fig, ax = plt.subplots(1, 1, figsize=np.array((6.4, 4.8))*figscale)
ax.loglog(a_list, Δt_to_Δa(Δt_dynamical), 'C0-', label=r'dynamical')

ax.loglog(a_list, Δt_to_Δa(Δt_Δa_early), 'C1--')
mask = (Δt_Δa_early == Δt_hubble_used)
ax.loglog(a_list[mask], Δt_to_Δa(Δt_Δa_early)[mask], 'C1-', label=r'$\Delta a$ (early)')

ax.loglog(a_list, Δt_to_Δa(Δt_Δa_late), 'C2-', label=r'$\Delta a$ (late)')

ax.loglog(a_list, Δt_to_Δa(Δt_hubble), 'C3--')
mask = (Δt_hubble == Δt_hubble_used)
ax.loglog(a_list[mask], Δt_to_Δa(Δt_hubble)[mask], 'C3-', label=r'$H^{-1}$')

mask = (a_list > -np.inf)
for i in (0, 10, 19):
    mask[i] = False
ax.loglog(a_list[mask], Δt_to_Δa(Δt_p3m)[mask], 'C4-', label='P$^3$M')

ax.loglog(a_list, Δt_to_Δa(Δt_max), 'k:')

ax.legend()
ax.set_xlim(5e-3, 1)
ax.set_ylim(8e-4, 0.04) #4.3)
ax.set_xlabel('$a$')
ax.set_ylabel('$\Delta a$')
fig.tight_layout()
plt.savefig('Deltaa.pdf')
plt.close()
"""



##############
# Delta loga #
##############
f = scipy.interpolate.interp1d(np.log(t_list), np.log(a_list), kind='linear', fill_value='extrapolate')
def t_to_a(t):
    return np.exp(f(np.log(t)))
def Δt_to_Δloga(Δt):
    return np.log(t_to_a(t_list + Δt)) - np.log(a_list)

#fig, ax = plt.subplots(1, 1, figsize=np.array((6.4, 4.8))*figscale)
fig, ax = plt.subplots(1, 1, figsize=(width, height))

ax.loglog(a_list, Δt_to_Δloga(Δt_dynamical), 'C0-', label=r'dynamical')

ax.loglog(a_list, Δt_to_Δloga(Δt_Δa_early), 'C1--', alpha=0.7)
mask = (Δt_Δa_early == Δt_hubble_used)
ax.loglog(a_list[mask], Δt_to_Δloga(Δt_Δa_early)[mask], 'C1-', label=r'$\Delta a$ (early)')

ax.loglog(a_list, Δt_to_Δloga(Δt_hubble), 'C1--', alpha=0.7)
mask = (Δt_hubble == Δt_hubble_used)
mask[-1] = True
ax.loglog(a_list[mask], Δt_to_Δloga(Δt_hubble)[mask], 'C1-', label=r'$H^{-1}$')

mask = (a_list > -np.inf)
#for i in (0, 10, 19, -3):
#    mask[i] = False
ax.loglog(a_list[mask], Δt_to_Δloga(Δt_p3m)[mask], 'C2-', label='P$^3$M')

# Extra P³M
x_extras = []
y_extras = []
for box, d in extra.items():
    if box == 64:
        continue
    f_extra = scipy.interpolate.interp1d(np.log(t_list), np.log(a_list), kind='linear', fill_value='extrapolate')
    def t_to_a_extra(t):
        return np.exp(f_extra(np.log(t)))
    def Δt_to_Δloga_extra(Δt):
        return np.log(t_to_a_extra(d['t'] + Δt)) - np.log(d['a'])
    mask_extra = (d['a'] > -np.inf)
    #for i in (0, 10, 19, -3):
    #    mask_extra[i] = False
    _x = d['a'][mask_extra]
    _y = Δt_to_Δloga_extra(d['Δt_p3m'])[mask_extra]
    if box == 128:
        _y[:17] *= 0.98
    ax.loglog(_x, _y, 'C2-')
    x_extras.append(_x)
    y_extras.append(_y)
ax.loglog(a_list, Δt_to_Δloga(Δt_Δa_late), 'C3-', label=r'$\Delta a$ (late)')

# Constant Δt
for n_times in np.logspace(np.log10(14), np.log10(7e+4), 5):
    n_times = int(round(n_times))
    times = np.linspace(t_list[0]*(1 + 1e-6), t_list[-1], n_times)
    a_vals = np.array([t_to_a(_t) for _t in times])
    x = a_vals
    y = np.diff(np.log(a_vals))
    yy = np.exp(np.interp(np.linspace(0, 1, len(y)+1), np.linspace(0, 1, len(y)), np.log(y)))
    ax.loglog(x, yy, 'k--', lw=0.5, zorder=-np.inf)

# Black dots
x = list(a_list).copy()
y = list(Δt_to_Δloga(Δt_max))
y = y[:10] + [y[9]] + y[10:]
x = x[:10] + [0.5*(x[9] + x[10])] + x[10:]
x[-1] *= 1.02
ax.loglog(x, y, color='k',
    linestyle=(0, (0, 2.57)),
    dash_capstyle='round',
)
# Black dots for ther green lines
for j, (x_ext, y_ext) in enumerate(zip(x_extras, y_extras)):
    x = np.array(x_ext).copy()
    y = np.array(y_ext).copy()
    i = np.argmin((np.log(x) - np.log(0.1))**2)
    mask = (y <= Δt_to_Δloga(Δt_hubble[i])[i])
    count = {0: 1, 1: 2}[j]
    for i in range(mask.size):
        if count == 0:
            break
        if mask[i]:
            mask[i] = False
            count -= 1
    ax.loglog(x[mask], y[mask], color='k', zorder=np.inf,
        linestyle=(0, (0, 2.57)),
       dash_capstyle='round',
    )

#ax.legend()
ax.set_xlim(5e-3, 1)
ax.set_ylim(1.0e-2, 0.3) #4.3)
ax.set_xlabel('$a$')
ax.set_ylabel('$\Delta \ln a$')
ax.set_xticks([0.01, 0.1, 1])
ax.set_xticklabels(['$0.01$', '$0.1$', '$1$'])
ax.set_yticks([0.01, 0.1])
ax.set_yticklabels(['$0.01$', '$0.1$'])

X, Y = 0.817, 0.860
sizex = 0.24*0.3
sizey = 0.083*0.85
q = 12.0
qq = 18
XX = np.array([X - 0.5*sizex, X + 0.5*sizex, X + 0.5*sizex, X - 0.5*sizex, X - 0.5*sizex])
YY = np.array([Y - 0.5*sizey, Y - 0.5*sizey, Y + 0.5*sizey, Y + 0.5*sizey, Y - 0.5*sizey])
meanXX = X
meanYY = Y
XX -= meanXX
YY -= meanYY
XX, YY = np.array([
    [np.cos(qq*np.pi/180), -np.sin(qq*np.pi/180)],
    [np.sin(qq*np.pi/180), np.cos(qq*np.pi/180)]
]) @ np.array([XX, YY])
XX += meanXX
YY += meanYY
ax.fill(XX, YY, 'w', alpha=0.35, ec='none', transform=ax.transAxes, zorder=100)
ax.text(X, Y, r'dynamical',
    ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, zorder=101,
    rotation=q)
ax.text(0.325, 0.572, r'$\Delta a$ (early)',
    ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, rotation=-43.0)
ax.text(0.828, 0.572, r'$\Delta a$ (late)',
    ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, rotation=-43.0)

X, Y = 0.550, 0.380
sizex = 0.16
sizey = 0.083
ax.text(X, Y, r'Hubble',
    ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, zorder=101)
ax.fill(
    [X - 0.5*sizex, X + 0.5*sizex, X + 0.5*sizex, X + 0.3*sizex, X - 0.5*sizex, X - 0.5*sizex],
    [Y - 0.4*sizey, Y - 0.4*sizey, Y + 0.0*sizey, Y + 0.5*sizey, Y + 0.5*sizey, Y - 0.4*sizey],
    'w', alpha=0.35, ec='none', transform=ax.transAxes, zorder=100,
)

X, Y = 0.5635, 0.572
sizex = 0.16
sizey = 0.083
theta = -43.0
ax.text(X, Y, r'$\text{\PTHREEM{}}$',
    ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, rotation=theta, zorder=101)
i_bgn, i_end = 61, 75
ax.plot(x_extras[0][i_bgn:i_end], y_extras[0][i_bgn:i_end], 'w-', alpha=0.35, lw=2.5, zorder=100)

# Matter-lambda equivalence
cosmo = Class()
cosmo.set({
    'H0': 67,
    'Omega_b': 0.049,
    'Omega_cdm': 0.27,
})
cosmo.compute()
bg = cosmo.get_background()
index = np.argmin((np.log(bg['(.)rho_lambda']) - np.log(bg['(.)rho_b'] + bg['(.)rho_cdm']))**2)
a_mL = 1/(1 + bg['z'][index])
ax.plot([a_mL]*2, [0.001, 1], '--k', lw=0.5, zorder=-np.inf)
index = np.argmin((np.log(bg['(.)rho_g'] + bg['(.)rho_ur']) - np.log(bg['(.)rho_b'] + bg['(.)rho_cdm']))**2)
a_rm = 1/(1 + bg['z'][index])

# Save
fig.subplots_adjust(wspace=0, hspace=0) #, left=0.15, right=0.98, bottom=0.18, top=0.975)
cropsave(fig, '../figure/timestepsize.pdf')

