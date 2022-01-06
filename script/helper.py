import collections, os, pickle, re, shutil, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate



"""
This script provide common functionality to the other Python scripts.

Do not wory about the various references to directories in /home/jeppe/...
This is just a relic of the creation of these scripts,
no such paths need to exist.
"""



# Directories
grendel_dir = '/home/jeppe/mnt/grendel'
log_dir = f'{grendel_dir}/log/concept'
memory_dir  = f'{grendel_dir}/mem'
t_a_filename = f'{grendel_dir}/t_a.dat'
cache_dir = os.path.dirname(__file__) + '/../data'

# Saving with automatic margins
def cropsave(fig, fname, pad=0.01, func=None):
    import subprocess, numpy as np
    dname = os.path.dirname(fname)
    frac = 0.3  # magic constant (should never have to be changed)
    plt.figure(fig.number)
    plt.draw()
    xticks = []
    yticks = []
    xticklabels = []
    yticklabels = []
    xlims = []
    ylims = []
    for ax in fig.axes:
        xticks.append(ax.get_xticks())
        yticks.append(ax.get_yticks())
        xticklabels.append(ax.get_xticklabels())
        yticklabels.append(ax.get_yticklabels())
        xlims.append(ax.get_xlim())
        ylims.append(ax.get_ylim())
    def set_ticksandlims():
        for i, ax in enumerate(fig.axes):
            ax.set_xticks(xticks[i])
            ax.set_yticks(yticks[i])
            ax.set_xlim(xlims[i])
            ax.set_ylim(ylims[i])
    def measure():
        set_ticksandlims()
        plt.savefig(f'{dname}/.tmp.pdf')
        size, size_cropped = [
            tuple(map(float, line.split()[2::2]))
            for line in
            subprocess.Popen(
                f'cd "{dname}" '
                f'&& pdfcrop --hires .tmp.pdf .tmp_cropped.pdf >/dev/null '
                f'&& (pdfinfo .tmp.pdf && pdfinfo .tmp_cropped.pdf) '
                f'| grep "Page size"',
                shell=True, stdout=subprocess.PIPE,
            ).communicate()[0].decode().strip().split('\n')
        ]
        return size_cropped[0]/size[0], size_cropped[1]/size[1]
    fig.subplots_adjust(
        left=frac, right=(1 + frac),
        bottom=frac, top=(1 - frac),
    )
    fac_left, _ = measure()
    fig.subplots_adjust(
        left=frac, right=(1 - frac),
        bottom=frac, top=(1 + frac),
    )
    _, fac_bottom = measure()
    fig.subplots_adjust(
        left=-frac, right=(1 - frac),
        bottom=frac, top=(1 - frac),
    )
    fac_right, _ = measure()
    fig.subplots_adjust(
        left=frac, right=(1 - frac),
        bottom=-frac, top=(1 - frac),
    )
    _, fac_top = measure()
    subprocess.Popen(f'cd "{dname}" && rm -f .tmp.pdf .tmp_cropped.pdf', shell=True)
    left = frac + fac_left - 1
    right = 2 - frac - fac_right
    bottom = frac + fac_bottom - 1
    top = 2 - frac - fac_top
    width = right - left
    height = top - bottom
    fig.subplots_adjust(
        left=(left     + pad/2*width),
        right=(right   - pad/2*width),
        bottom=(bottom + pad/2*height),
        top=(top       - pad/2*height),
    )
    set_ticksandlims()
    if func is not None:
        func()
    plt.savefig(fname)

# Function for taking the running average over a period
# of 8 steps, as well as replacing the first and last
# step with linear extrapolations.
def mean8(y, ignore_bgn=2, period=8, *, n_steps):
    #if len(y) != n_steps:
    #    print(f'Passed y to mean8() with length {len(y)} != {n_steps}', file=sys.stderr)
    #    sys.exit(1)
    y = np.array(y).copy()
    for i in list(range(ignore_bgn))[::-1]:
        x = np.arange(i + 1, i + period + 1)
        a, b = np.polyfit(x, y[x], deg=1)
        y[i] = a*i + b
    y_mean8 = []
    for i in range(len(y) - 2):
        i_left = max(i - period//2, 1)
        i_right = min(i_left + period, len(y) - 1)
        i_left -= period - (i_right - i_left)
        if i_left < 0:
            i_left = 0
        y_mean8.append(np.mean(y[i_left:i_right]))
    for i in range(period//2, -1, -1):
        x = np.arange(i + 1, period)
        a, b = np.polyfit(x, np.array(y_mean8)[x], deg=1)
        y_mean8[i] = a*i + b
    for i in (2, 1):
        x = np.arange(n_steps - n_steps%period, n_steps - i)
        if len(x) == 0:
            break
        a, b = np.polyfit(x, np.array(y_mean8)[x], deg=1)
        y_mean8.append(a*(n_steps - i) + b)
    x = np.arange(1, period + 1)
    a, b = np.polyfit(x, np.array(y_mean8)[x], deg=1)
    y_mean8[0] = a*0 + b
    y_mean8 = np.array(y_mean8)
    return y_mean8

def get_factor_after_symplectifying():
    """PR247 (https://github.com/AarhusCosmology/concept/pull/247)
    made a change to the time-integration, improving the symplecticity.
    This came with a slight drop in performance, accounted for here.
    """
    fac = 1.016  # 1.6 % performanec hit
    return fac

# Primary function, loading in all data
def load(output_dir, sim, check_spectra=True, cache_assume_uptodate=False):
    # Load pickled data if present
    pickle_filename = (
        '../.pickle/'
        + output_dir
            .replace('/home/jeppe/mnt/grendel2/', '')
            .replace('/home/jeppe/mnt/grendel/', '')
            .replace('/', '_').replace(' ', '_')
        + f'_{sim}.pkl'
    )
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            infos = pickle.load(f)
        set_system_time2time_step(infos)
        return infos
    # Get mapping from nprocs -> job ID's
    jobs = get_jobs(output_dir, sim, check_spectra, cache_assume_uptodate)
    # Extract data from log files
    infos = {}
    for nprocs, jobid in jobs.items():
        data, computation_times, t_total = parse_log(jobid, cache_assume_uptodate)
        info = {
            'data': data,
            'computation_times': computation_times,
            't_total': t_total,
        }
        infos[nprocs] = info
    # Read in memory consumption
    set_system_time2time_step(infos)
    for nprocs, jobid in jobs.items():
        info = infos[nprocs]
        info['mem'] = read_mem(
            jobid, info['system_time2time_step'], cache_assume_uptodate,
        )
    # Dump to file and return
    del_system_time2time_step(infos)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(infos, f, pickle.HIGHEST_PROTOCOL)
    set_system_time2time_step(infos)
    return infos

def load_gadget(output_dir, gadget_log_dir, check_spectra=True, cache_assume_uptodate=False):
    sim = 'gadget'
    # Load pickled data if present
    pickle_filename = (
        '../.pickle/gadget_'
        + output_dir
            .replace('/home/jeppe/mnt/grendel2/', '')
            .replace('/home/jeppe/mnt/grendel/', '')
            .replace('/', '_').replace(' ', '_')
        + f'_{sim}.pkl'
    )
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            infos = pickle.load(f)
        return infos
    # Get mapping from nprocs -> job ID's
    jobs = get_jobs(output_dir, 'Gadget2', check_spectra, cache_assume_uptodate)
    # Load mapping a -> t/Gyr
    t_a_filename_cached = cache_grendel(t_a_filename, cache_assume_uptodate)
    t, a = np.loadtxt(t_a_filename_cached, unpack=True)
    loga_logt = scipy.interpolate.interp1d(
            np.log(a),
            np.log(t),
            kind='cubic',
    )
    def get_cosmic_time(a):
        return np.exp(loga_logt(np.log(a)))
    # Extract data from log files
    infos = {}
    for nprocs, jobid in jobs.items():
        print(f'Parsing log {jobid} ...', end='', flush=True)
        filename = cache_grendel(f'{gadget_log_dir}/{jobid}', cache_assume_uptodate)
        with open(filename) as f:
            lines = f.readlines()
        data = []
        at_new_timestep = False
        for line in lines:
            if line.startswith('Begin Step '):
                at_new_timestep = True
                match = re.search(r'Begin Step (.*?), Time: (.*?),', line)
                time_step = int(match.group(1))
                scale_factor = float(match.group(2))
                cosmic_time = get_cosmic_time(scale_factor)
            if line.startswith('System time: '):
                system_time = float(line.strip().split()[-1])
                if at_new_timestep:
                    log_data = LogData(
                        time_step,
                        system_time,
                        scale_factor,
                        cosmic_time,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    data.append(log_data)
                at_new_timestep = False
        t_total = system_time - data[0].system_time
        computation_times = []
        for log_data in reversed(data):
            computation_times.append(system_time - log_data.system_time)
            system_time = log_data.system_time
        computation_times = np.array(computation_times[::-1])
        info = {
            'data': data,
            'computation_times': computation_times,
            't_total': t_total,
        }
        infos[nprocs] = info
    # Read in memory consumption
    for nprocs, jobid in jobs.items():
        info = infos[nprocs]
        info['mem'] = read_mem(jobid, None, cache_assume_uptodate)
    # Dump to file and return
    with open(pickle_filename, 'wb') as f:
        pickle.dump(infos, f, pickle.HIGHEST_PROTOCOL)
    return infos

# Functions for constructing and removing
# spline mappings from system time to time step.
def set_system_time2time_step(infos):
    for info in infos.values():
        data = info['data']
        n_steps = len(data)
        f = scipy.interpolate.interp1d(
            [data[i].system_time for i in range(n_steps)],
            [data[i].time_step for i in range(n_steps)],
            kind='linear',
            bounds_error=False,
            fill_value=(data[0].time_step, data[-1].time_step),
        )
        info['system_time2time_step'] = (lambda system_time, f=f: np.round(f(system_time)))
def del_system_time2time_step(infos):
    for info in infos.values():
        info.pop('system_time2time_step', None)

# Function for caching a file. It will copy the file
# from the mounted drive if its time stamp is newer
# than the local version. The path to the local file
# will be returned.
def cache_grendel(filename_grendel, cache_assume_uptodate):
    filename_grendel_bare = filename_grendel[len(grendel_dir)+1:]
    filename = f'{cache_dir}/{filename_grendel_bare}'
    if cache_assume_uptodate:
        return filename
    if not filename_grendel.startswith(grendel_dir):
        print(
            f'cache_grendel() called with non-Grendel path "{filename_grendel}"',
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(filename_grendel):
        if not glob(f'{grendel_dir}/*'):
            msg = 'It looks like Grendel is not mounted'
        elif os.path.isdir(filename_grendel):
            msg = f'cache_grendel() called with *directory* "{filename_grendel}"'
        else:
            msg = f'cache_grendel() called with non-existing "{filename_grendel}"'
        print(msg, file=sys.stderr)
        sys.exit(1)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(filename):
        if os.path.getmtime(filename_grendel) > os.path.getmtime(filename):
            print(f'Updating cached "{filename_grendel_bare}"', flush=True)
            shutil.copy(filename_grendel, filename)
    else:
        print(f'Caching "{filename_grendel_bare}"', flush=True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copy(filename_grendel, filename)
    return filename

# Function for finding which simulations have been run.
# A dict {jobid: nprocs} will be returned.
def get_jobs(output_dir, sim, check_spectra, cache_assume_uptodate):
    print(f'Searching for available nprocs of "{sim}" ...', end='', flush=True)
    jobs = {}
    nprocs_all = []
    powerspec_ref = None
    lookup = f'{output_dir}/{sim}/*'
    if cache_assume_uptodate:
        lookup = f'{cache_dir}/' + lookup[len(grendel_dir)+1:]
    for dirname in glob(lookup):
        try:
            nprocs = int(os.path.basename(dirname))
        except ValueError:
            continue
        if nprocs in ():
            continue
        filename = f'{dirname}/powerspec_a=1.00'
        if not os.path.exists(filename):
            filename = f'{dirname}/powerspec_snapshot_000'
            if not os.path.exists(filename):
                continue
        if sim.startswith('Gadget'):
            if not cache_assume_uptodate:
                for aux in [f'{dirname}/powerspec_snapshot_000', f'{dirname}/jobid']:
                    if os.path.exists(aux):
                        cache_grendel(aux, cache_assume_uptodate)  # just for caching
            if not os.path.exists(f'{dirname}/powerspec_snapshot_000') or not os.path.exists(f'{dirname}/jobid'):
                print(f'Skipping "{dirname}"')
                continue
        if not cache_assume_uptodate:
            filename = cache_grendel(filename, cache_assume_uptodate)
        nprocs_all.append(nprocs)
        # Consistency check
        powerspec_data = np.loadtxt(filename)
        if powerspec_ref is None:
            # For some reason, the Gadget2 nprocs = 1 sim does not
            # quite give the same results!
            if not ('gadget2' in f'{output_dir}/{sim}'.lower() and nprocs == 1):
                powerspec_ref = powerspec_data
        #if (powerspec_ref is not None
        #    and check_spectra and not np.allclose(
        #            powerspec_data, powerspec_ref, rtol=1e-2, atol=0,
        #        )
        #):
        #    print(f'Inconsistent data in "{filename}"', file=sys.stderr)
        #    sys.exit(1)
        # Get jobid
        if sim.startswith('Gadget'):
            with open(f'{dirname}/jobid') as f:
                jobid = int(f.read().strip())
        else:
            with open(filename) as f:
                header, *lines = f.readlines()
            jobid = int(re.search(r' job (\d+) ', header).group(1))
        jobs[nprocs] = jobid
    nprocs_all.sort()
    jobs = {nprocs: jobs[nprocs] for nprocs in nprocs_all}
    print(f' done\n    Found', ', '.join(map(str, nprocs_all)), flush=True)
    return jobs

# Function for parsing a CONCEPT log file.
# The return value is a list of named tuples,
# each corresponding to a time step.
def parse_log(jobid, cache_assume_uptodate):
    print(f'Parsing log {jobid} ...', end='', flush=True)
    filename = cache_grendel(f'{log_dir}/{jobid}', cache_assume_uptodate)
    # Read in
    with open(filename) as f:
        lines = f.readlines()
    # Remove formatting
    lines = [re.sub(chr(27) + r'\[\d+m', '', line).rstrip() for line in lines]
    # Parse
    def to_seconds(t):
        t = t.strip()
        if ',' in t:
            days, t = t.split(',')
            days = int(days.replace('days', '').replace('day', '').strip())
            return 24*60*60*days + to_seconds(t)
        if t.count(':') == 2:
            h, m, s = [int(el) for el in t.split(':')]
            return float(h*60**2 + m*60 + s)
        elif t.count(':') == 1:
            m, s = [int(el) for el in t.split(':')]
            return float(m*60 + s)
        elif t.endswith(' s'):
            return float(t.rstrip(' s'))
        elif t.endswith(' ms'):
            return float(t.rstrip(' ms'))/1000
        print(f'Do not know how to convert "{t}" to seconds', file=sys.stderr)
        sys.exit(1)
    time_step = -1
    t_subtract = 0
    t_total = -1
    subtiling = []
    data = []
    for i, line in enumerate(lines):
        if (match := re.search(r'Domain decomposition: (.+)', line)):
            nprocs = np.prod([int(cut.strip()) for cut in match.group(1).split('×')])
        if (match := re.search(r'Subtile decomposition.+:(.+)', line)):
            subtiling_mean = np.mean([int(cut.strip()) for cut in match.group(1).split('×')])
            if np.isclose(subtiling_mean, int(subtiling_mean)):
                subtiling_mean = int(subtiling_mean)
            subtiling = np.array([subtiling_mean]*nprocs, dtype=float)
        if (match := re.search(r'^Time step +(\d+)', line)):
            time_step = int(match.group(1))
            # New time step
            t_shortrange = 0
            t_longrange = 0
            t_fft = 0
        if time_step < 0:
            continue
        if (match := re.search(r'^System time: +(\d+\.\d+)', line)):
            system_time = float(match.group(1))
        elif (match := re.search(r'^Scale factor: +(\d+\.\d+)', line)):
            scale_factor = float(match.group(1))
        elif (match := re.search(r'^Cosmic time: +(\d+\.\d+)', line)):
            cosmic_time = float(match.group(1))
        elif (match := re.search(r'^Step size: +(\d+\.\d+)', line)):
            step_size = float(match.group(1))
        elif (match := re.search(r'^ +GADGET halo: +(.+)', line)):
            rung_population = np.array(eval('[' + match.group(1) + ']'))
        elif (match := (
               re.search(r'^ +\.\.\. +\((.+)\)', line)
            or re.search(r'^\(short-range only\) \.\.\. +\((.+)\)', line)
        )):
            t_shortrange += to_seconds(match.group(1))
        elif line.strip() == '...':
            for line_next in lines[i+1:]:
                if line_next.startswith(' '*20) and (match := re.search(r'^ +\((.+)\)', line_next)):
                    t_shortrange += to_seconds(match.group(1))
                    break
        elif line.startswith('(long-range only)'):
            other_line = lines[i + 12]
            t_longrange += to_seconds(re.search(r'^ +\((.+)\)', other_line).group(1))
        elif (match := re.search(
            r'^ +Transforming to real space potential \.\.\. +\((.+)\)', line,
        )):
            t_fft += 2*to_seconds(match.group(1))
        elif (match := re.search(
            r'Rank (.+): Refined subtile decomposition \(gravity\): (.+)', line,
        )):
            rank = int(match.group(1))
            subtiling_mean = np.mean([int(cut.strip()) for cut in match.group(2).split('×')])
            if np.isclose(subtiling_mean, int(subtiling_mean)):
                subtiling_mean = int(subtiling_mean)
            subtiling[rank] = subtiling_mean
        elif line.startswith('Load imbalance:'):
            load_imbalance = []
            for other_line in lines[i+1:]:
                if not other_line.startswith('    Process '):
                    break
                load_imbalance.append(float(
                    re.search(r' ([+-].+)%', other_line).group(1).replace(' ', '')
                )/100)
            load_imbalance = np.array(load_imbalance)
            # End of time step
            t_fft = min(t_fft, 0.9*t_longrange)
            log_data = LogData(
                time_step,
                system_time,
                scale_factor,
                cosmic_time,
                step_size,
                rung_population,
                t_shortrange,
                t_longrange,
                t_fft,
                load_imbalance,
                subtiling.copy(),
            )
            data.append(log_data)
        elif 'powerspec_a=1.00" ...' in line or 'Computing power spectrum' in line:
            match = re.search(r' +\((.+)\)', line)
            if match is not None:
                t_subtract += to_seconds(match.group(1))
        elif (match := re.search(r'^ *Total execution time: +(.+)', line)):
            t_total = to_seconds(match.group(1))
        elif line.startswith('Drifting GADGET halo') and False:  # only for debugging
            # End of time step
            log_data = LogData(
                time_step,
                system_time,
                scale_factor,
                cosmic_time,
                step_size,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            data.append(log_data)
    # Final system time not counting powerspec output
    system_time -= t_subtract
    # Total execution time not counting I/O
    t_total_true = 0
    if data:
        t_total_true = system_time - data[0].system_time
        if t_total_true > t_total:
            print(
                f'Total execution time without I/O ({t_total_true} s) supposedly larger '
                f'than the full computation time ({t_total} s)',
                flush=True, file=sys.stderr,
            )
            sys.exit(1)
    t_total = t_total_true
    # All lines parsed. Determine computation time per step.
    computation_times = []
    for log_data in reversed(data):
        computation_times.append(system_time - log_data.system_time)
        system_time = log_data.system_time
    computation_times = np.array(computation_times[::-1])
    print(' done', flush=True)
    return data, computation_times, t_total
LogData = collections.namedtuple(
    'LogData',
    (
        'time_step',        # time step number
        'system_time',      # system time [s]
        'scale_factor',     # scale factor
        'cosmic_time',      # cosmic time [Gyr]
        'step_size',        # time step size [Gyr]
        'rung_population',  # rung population of particle component
        't_shortrange',     # time spent in short-range computation [s]
        't_longrange',      # time spent in long-range computation [s]
        't_fft',            # time spent in FFT part of long-range computation [s]
        'load_imbalance',   # load imbalance of each process as fraction (not percent)
        'subtiling',        # subtiling (assumed cubic)
    ),
)

# Function for reading the memory dump of a job
def read_mem(jobid, system_time2time_step, cache_assume_uptodate):
    print(f'Reading memory dump of {jobid} ...', end='', flush=True)
    dirname = f'{memory_dir}/{jobid}'
    if cache_assume_uptodate:
        dirname = f'{cache_dir}/' + dirname[len(grendel_dir)+1:]
    filenames = glob(f'{dirname}/*')
    if not filenames:
        print(' none found')
        return
    i_nodes = set()
    for filename in filenames:
        match = re.search(r'^(\d+)_\d+', os.path.basename(filename))
        if not match:
            continue
        i_nodes.add(int(match.group(1)))
    i_nodes = sorted(i_nodes)
    n_nodes = len(i_nodes)
    if i_nodes != list(range(n_nodes)):
        print('Weird filenames in "{dirname}"', file=sys.stderr)
        sys.exit(1)
    filenames_sorted = []
    for filename in filenames:
        match = re.search(rf'^0_(\d+)', os.path.basename(filename))
        if not match:
            continue
        filenames_sorted.append((float(match.group(1)), filename))
    filenames_sorted = [_[1] for _ in sorted(filenames_sorted)]
    # Use cache
    for j, filename in enumerate(filenames_sorted):
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        for i in range(n_nodes):
            filename_i = f'{dirname}/{i}_' + basename.split('_')[-1]
            # Check for existing and non-empty file
            if cache_assume_uptodate:
                #filename_i = cache_grendel(filename_i, cache_assume_uptodate)
                if not os.path.exists(filename_i):
                    # File does not exist. Remove for all nodes.
                    filenames_sorted[j] = None
                    continue
            else:
                if not os.path.exists(filename_i):
                    # File does not exist. Remove for all nodes.
                    filenames_sorted[j] = None
                    continue
                else:
                    filename_i = cache_grendel(filename_i, cache_assume_uptodate)
            with open(filename_i, 'r') as f:
                content = f.read()
            if not content:
                # Empty file. Remove for all nodes.
                filenames_sorted[j] = None
                continue
    filenames_sorted = [filename for filename in filenames_sorted if filename is not None]
    if not cache_assume_uptodate:
        filenames_sorted = [
            cache_grendel(filename, cache_assume_uptodate)
            for filename in filenames_sorted
        ]
    # Read in
    VSZ_all, RSS_all, PSS_all = [], [], []
    system_time_all = []
    for filename in filenames_sorted:
        match = re.search(rf'^0_(\d+)', os.path.basename(filename))
        if not match:
            continue
        system_time = float(match.group(1))
        system_time_all.append(system_time)
        VSZ, RSS, PSS = [], [], []
        for i_node in range(n_nodes):
            filename_node = (
                os.path.dirname(filename) + '/'
                + os.path.basename(filename).replace('0_', f'{i_node}_')
            )
            VSZ_node, RSS_node, PSS_node = np.loadtxt(filename_node, unpack=True)
            if VSZ_node.ndim == 0:
                VSZ_node = [VSZ_node]
                RSS_node = [RSS_node]
                PSS_node = [PSS_node]
            VSZ += list(VSZ_node)
            RSS += list(RSS_node)
            PSS += list(PSS_node)
        VSZ_all.append(np.array(VSZ))
        RSS_all.append(np.array(RSS))
        PSS_all.append(np.array(PSS))
    system_time_all = np.array(system_time_all)
    # If no system_time2time_step is passed, return mem as is,
    # without mapping to time steps.
    if system_time2time_step is None:
        # Remove spikes
        nprocs = len(VSZ_all[0])
        for MEM in (VSZ_all, RSS_all, PSS_all):
            for rank in range(nprocs):
                for i in range(1, len(VSZ_all) - 1):
                    if MEM[i - 1][rank] < MEM[i][rank] > MEM[i + 1][rank]:
                        MEM[i][rank] = 0.5*(MEM[i - 1][rank] + MEM[i + 1][rank])
        # Transpose data
        mem = {'VSZ': [], 'RSS': [], 'PSS': []}
        for rank in range(nprocs):
            mem['VSZ'].append([VSZ[rank] for VSZ in VSZ_all])
            mem['RSS'].append([RSS[rank] for RSS in RSS_all])
            mem['PSS'].append([PSS[rank] for PSS in PSS_all])
        mem['VSZ'] = np.array(mem['VSZ'])
        mem['RSS'] = np.array(mem['RSS'])
        mem['PSS'] = np.array(mem['PSS'])
        print(' done', flush=True)
        return mem
    # Map to time steps
    VSZ_timesteps = {}
    RSS_timesteps = {}
    PSS_timesteps = {}
    for i in range(len(system_time_all)):
        time_step = system_time2time_step(system_time_all[i])
        if time_step in VSZ_timesteps:
            if time_step > 50:
                continue
            if time_step > 20 and np.mean(PSS_all[i]) > np.mean(PSS_timesteps[time_step]):
                continue
            if time_step <= 20 and np.mean(PSS_all[i]) < np.mean(PSS_timesteps[time_step]):
                continue
        VSZ_timesteps[time_step] = VSZ_all[i]
        RSS_timesteps[time_step] = RSS_all[i]
        PSS_timesteps[time_step] = PSS_all[i]
    # Fill possibly missing values
    steps_existing = list(VSZ_timesteps.keys())
    for i in range(int(round(np.min(steps_existing))), int(round(np.max(steps_existing)))):
        if i not in steps_existing:
            i_prev = i - 1
            for i_next in range(i + 1, int(round(np.max(steps_existing)))):
                if i_next in steps_existing:
                    break
            else:
                i_next = -1
            if i_prev > -1 and i_next > -1:
                weight_prev = 1/(i - i_prev)
                weight_next = 1/(i_next - i)
                VSZ_timesteps[i] = (
                    weight_prev*VSZ_timesteps[i_prev] + weight_next*VSZ_timesteps[i_next]
                )/(weight_prev + weight_next)
                RSS_timesteps[i] = (
                    weight_prev*RSS_timesteps[i_prev] + weight_next*RSS_timesteps[i_next]
                )/(weight_prev + weight_next)
                PSS_timesteps[i] = (
                    weight_prev*PSS_timesteps[i_prev] + weight_next*PSS_timesteps[i_next]
                )/(weight_prev + weight_next)
            elif i_prev > -1:
                VSZ_timesteps[i] = VSZ_timesteps[i_prev]
                RSS_timesteps[i] = RSS_timesteps[i_prev]
                PSS_timesteps[i] = PSS_timesteps[i_prev]
            elif i_next > -1:
                VSZ_timesteps[i] = VSZ_timesteps[i_next]
                RSS_timesteps[i] = RSS_timesteps[i_next]
                PSS_timesteps[i] = PSS_timesteps[i_next]
            else:
                print('Error filling missing mem values', flush=True, file=sys.stderr)
                sys.exit(1)
    # Re-order
    time_steps = sorted(VSZ_timesteps.keys())
    VSZ_timesteps = [VSZ_timesteps[time_step] for time_step in time_steps]
    RSS_timesteps = [RSS_timesteps[time_step] for time_step in time_steps]
    PSS_timesteps = [PSS_timesteps[time_step] for time_step in time_steps]   
    # Remove spikes
    nprocs = len(VSZ_timesteps[0])
    for MEM in (VSZ_timesteps, RSS_timesteps, PSS_timesteps):
        for rank in range(nprocs):
            for i in range(1, len(time_steps) - 1):
                if MEM[i - 1][rank] < MEM[i][rank] > MEM[i + 1][rank]:
                    MEM[i][rank] = 0.5*(MEM[i - 1][rank] + MEM[i + 1][rank])
    # Transpose data
    mem = {'VSZ': [], 'RSS': [], 'PSS': []}
    for rank in range(nprocs):
        mem['VSZ'].append([VSZ[rank] for VSZ in VSZ_timesteps])
        mem['RSS'].append([RSS[rank] for RSS in RSS_timesteps])
        mem['PSS'].append([PSS[rank] for PSS in PSS_timesteps])
    mem['VSZ'] = np.array(mem['VSZ'])
    mem['RSS'] = np.array(mem['RSS'])
    mem['PSS'] = np.array(mem['PSS'])
    print(' done', flush=True)
    return mem    

