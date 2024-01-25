# Development
Setup conda env:
```sh
conda env create -f conda_environment.yml
conda activate bionomicon-crawler
```

To save dependencies:
```sh
conda env export | grep -v "^prefix: " > conda_environment.yml
```

## VSCODE
It is a good idea to setup vscode to use the conda env:
- Python: Select interpreter
Selecting the appropriate conda env