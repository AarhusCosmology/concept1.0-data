#####################
# Optional settings #
#####################
# Basic operation mode of code
OPT += -DPERIODIC

# Things that are always recommended
OPT += -DPEANOHILBERT
OPT += -DWALLCLOCK

# TreePM options
OPT += -DPMGRID=2048
OPT += -DASMTH=1.25
OPT += -DRCUT=4.5

# Time integration options
OPT += -DSYNCHRONIZATION

# Single/Double Precision
OPT += -DDOUBLEPRECISION

# Things for special behaviour
OPT += -DNOTYPEPREFIX_FFTW



########################################
# Settings for compilation and linking #
########################################
# Includes
fftw_incl = -I/path/to/fftw/include
gsl_incl = -I/path/to/gsl/include
mpi_incl = -I/path/to/mpi/include
includes = $(fftw_incl) $(gsl_incl) $(mpi_incl)

# Compiler options
CC = /path/to/mpi/bin/mpicc
other_cflags = -std=c99 -pthread -fPIC
optimizations = -DNDEBUG -O3 -funroll-loops -ffast-math -flto
CFLAGS += $(OPT) $(other_cflags) $(optimizations) $(includes)

# Libraries to link
fftw_libs = -L/path/to/fftw/lib -Wl,-rpath=/path/to/fftw/lib -lrfftw_mpi -lfftw_mpi -lrfftw -lfftw
gsl_libs = -L/path/to/gsl/lib -Wl,-rpath=/path/to/gsl/lib -lgsl -L/path/to/openblas/lib -Wl,-rpath=/path/to/openblas/lib -lopenblas -lm
mpi_libs = -L/path/to/mpi/lib -Wl,-rpath=/path/to/mpi/lib -lmpi
LDLIBS += $(fftw_libs) $(gsl_libs) $(mpi_libs)

# Linker options
comma = ,
optimizations_linker = -O3 -flto
LDFLAGS += $(optimizations) $(addprefix -Wl$(comma),$(optimizations_linker))



##############################
# Specification of filenames #
##############################
executable = Gadget2
sources =               \
    accel               \
    allocate            \
    allvars             \
    begrun              \
    density             \
    domain              \
    driftfac            \
    endrun              \
    forcetree           \
    global              \
    gravtree            \
    gravtree_forcetest  \
    hydra               \
    init                \
    io                  \
    longrange           \
    main                \
    ngb                 \
    peano               \
    pm_nonperiodic      \
    pm_periodic         \
    potential           \
    predict             \
    read_ic             \
    restart             \
    run                 \
    system              \
    timestep            \

headers =   \
    allvars \
    proto   \
    tags    \



###########
# Targets #
###########
# Link object files into executable
$(executable): $(addsuffix .o, $(sources))
	$(CC) $? -o $@ $(LDFLAGS) $(LDLIBS)

# Compile C source files into object files
$(addsuffix .o, $(sources)): %.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
$(addsuffix .o, $(sources)): $(addsuffix .h, $(headers)) $(MAKEFILE_LIST)

# Cleanup target
clean:
	rm -f $(addsuffix .o, $(sources)) $(executable)

