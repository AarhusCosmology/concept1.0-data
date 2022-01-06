python ?= python

all: figure

.PHONY: figure clean-figure clean-pickle

figure:
	@script/generate $(python)

clean-figure:
	$(RM) figure/* figure/.[^.]*

clean-pickle:
	$(RM) -r .pickle

clean: clean-figure clean-pickle

