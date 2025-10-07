.PHONY: tests notebooks

tests:
	uv run pytest

release: 
	uv build --wheel

notebooks: 
	uv run jupyter nbconvert --output-dir=docs/notebooks --to markdown notebooks/*.ipynb

build: 
	uv run jupyter nbconvert --execute --output-dir=docs/notebooks --to markdown notebooks/*.ipynb
