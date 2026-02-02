lint:
	pip install black isort
	isort .
	black .

install-requirements:
	pip install -r requirements.txt

create-venv:
	virtualenv -p python3.11 venv

create-requirements:
	pip install pipreqs
	pipreqs . --force

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/build

setup:
	pip install -e .

build:
	pip install build
	python -m build .
