PYTHON=python
SOURCE=source

.PHONY: test

test:
	${PYTHON} -m unittest discover -s test
