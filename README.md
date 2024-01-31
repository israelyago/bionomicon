# Overview
Our goal is to build a new AI model capable of learning from stablished biological data (Amino acid sequences) which sequences are enzymes and which aren't, i.e. we want a classifier in the format of f(amino_acid_sequence) -> enzyme / non-enzyme.

# Steps
1. Download data from Uniprot (FTP *recommended*) (uniprot_trembl & uniprot_sprot)
1. Extract data (as of writing, around 1.6TB of uncompressed data)
1. From the uncompressed data, we need to create an hdf5 file.
    1. Run extractor using `cargo` from rust. You need to run it 2 times, one for each uncompressed file (uniprot_trembl & uniprot_sprot)
    1. We need to merge these files and convert them to hdf5: Using python project parser
    1. From parser project, run it passing the uncompressed data file path and the output hdf5 file path
1. Now, with the hdf5 containing the information we need, we start building the AI model

## More information

This project is the implementation for the conclusion work entitled "[Automatic gene annotation with Artificial Intelligence: Binary classification between enzymes and non-enzymes](conclusion_work.pdf)" presented to the course of Biotechnology and Bioprocess Engineering as a partial requirement for obtaining the title of Biotechnology and Bioprocess Engineer (2024) for the Federal University of Technology - Paraná