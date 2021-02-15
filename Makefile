PYTHON=python

.PHONY: test

test:
	${PYTHON} -m unittest discover -v -s test
