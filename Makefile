
# Cut a PyPi release
pypi_release:
	python setup.py sdist bdist_wheel 
	twine check dist/* 
	twine upload --skip-existing dist/*

install:
	pip install .
