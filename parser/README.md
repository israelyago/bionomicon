# Parser

Converts the data extracted  in a .csv to a .hdf5 file to be used for training the AI model

## Development

Setup conda env:

```sh
conda env create -f conda_environment.yml
conda activate bionomicon-parser
```

In case the dependencies change, please run and save to git:

```sh
conda env export | grep -v "^prefix: " > conda_environment.yml
```

## Usage

```python
python main.py --input "PATH_TO_.CSV" --out "OUTPUT_HDF5_FILE_PATH" --logs "./logs" --release
```

Note: When runnig tests, omit the --release flag for faster processing. The data may not actually be written, because of the driver used in debug mode
Note: You may need to run two times. One for each generated .csv. The output .hdf5 file must be the same.

## VSCODE

It is a good idea to setup vscode to use the conda env:

- Python: Select interpreter
Selecting the appropriate conda env
