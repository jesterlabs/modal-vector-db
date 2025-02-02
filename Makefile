.PHONY: deploy

deploy:
	modal deploy embedders.py
	modal deploy vdb.py 