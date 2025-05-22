.PHONY: tests

tests:
	uv run pytest

release: 
	uv build --wheel
