#---- Basic operation mode of code
PERIODIC
#RANDOMIZE_DOMAINCENTER
GADGET2_HEADER

#---- Gravity calculation
SELFGRAVITY
#FMM
MULTIPOLE_ORDER=5

#---- TreePM Options
PMGRID=2048
ASMTH=1.25
RCUT=4.5

#---- Single/double precision and data types
DOUBLEPRECISION=1
DOUBLEPRECISION_FFTW

#---- Output/Input options
OUTPUT_NON_SYNCHRONIZED_ALLOWED

