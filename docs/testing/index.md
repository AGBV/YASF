# Unit tests
Testing is done using the built-in library `unittest`. To run all test files in the folder `tests/`, execute in the project root folder:
```sh
python -m unittest
```

# Coverage report
Install the `coverage` package:
```sh
pip install coverage
```
The command `coverage run` will replace any `python` command. To get a coverage report on the unit tests, run the following command:
```sh
coverage run -m unittest
```
To report this to [codacy](https://app.codacy.com/gh/AGBV/YASF), the `.coverage` file needs to be translated to the cobertura xml standard:
```sh
coverage xml -o cobertura.xml
```
Before submitting the report, provide an API key using an environmental variable `export CODACY_PROJECT_TOKEN=xxxxxxxxxxxxxxxxxxxxx`.
Now the report can be submitted to codacy using:
```sh
bash <(curl -Ls https://coverage.codacy.com/get.sh)
```
