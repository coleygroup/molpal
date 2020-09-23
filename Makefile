PYTHON=python

.PHONY: test

test:
	${PYTHON} -m unittest discover -s test
