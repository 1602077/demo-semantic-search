DEMO=embed.py

.DEFAULT_GOAL=help

.PHONY=demo
demo: # runs demo script
	@cd src/ && python ${DEMO}

.PHONY=env
env: # builds conda virtual environment
	@conda env create -f environment.yaml

.PHONY=help
help: # shows help message
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
