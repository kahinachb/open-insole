PYTHON ?= python

# Every number, table and figure of the paper, from the published evaluation tables (paper/evaluation).
# PREDICTIONS=<Full CV results root> also draws the per-task-family cycle figures; the predictions
# are not distributed with this repository.
paper-artefacts:
	$(PYTHON) paper/make_artefacts.py $(if $(PREDICTIONS),--predictions $(PREDICTIONS))

test:
	$(PYTHON) -m pytest paper/tests insole2grfm/tests tests

.PHONY: paper-artefacts test
