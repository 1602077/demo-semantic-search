DEMO=embed.py

.DEFAULT_GOAL=search

.PHONY=demo
demo: # runs demo script with default search query
	@cd src/ && python ${DEMO}

.PHONY=search
search: # runs demo script with specified search query
	@cd src/ && python ${DEMO} --search "$(search_query)"

.PHONY=env
env: # builds conda virtual environment
	@conda env create -f environment.yaml

.PHONY=help
help: # shows help message
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
