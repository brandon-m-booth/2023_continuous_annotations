# Robust Ground Truth from Continuous Annotations
This code base implements a novel method for fusing continuous annotations of a subjective construct to generate an accurate and reliable ground truth representation. The technique leverages traditional interval-scale continuous annotations and ordinal comparisons (i.e., A vs. B) to improve the validity of the construct representation by fixing several human-produced structured errors that tend to appear in traditional annotation paradigms when measuring the dynamics of a construct over time.

![Proposed Ground Truth Pipeline Image](https://github.com/brandon-m-booth/2023_continuous_annotations/blob/master/figures/png/proposed_ground_truth_pipeline.png?raw=true) <div style="text-align:center">**Figure**: An overview of the proposed *ordinal deformation* method</div>

This repository provides the figures, code, and results for the submitted publication:

<blockquote>
Brandon M. Booth, Shri S. Narayanan. "People Make Mistakes: Obtaining Accurate Ground Truth from Continuous Annotations of Subjective Constructs." Behavior Research Methods, 2023. [under review]
</blockquote>

Here is the version of the code and data in this repository released with the paper:

[![DOI](https://zenodo.org/badge/210796946.svg)](https://zenodo.org/badge/latestdoi/210796946)

## Folder Contents
1. *dataset* - Stores information about the movie violence dataset.  The movies themselves and movie clips are not included in this repository.
1. *figures* - Contains TeX (tikz), PNG, and [draw.io](https://app.diagrams.net/) formatted files used to generate figures in the paper
1. *results* - Holds annotation results for the main experiment (main\_study), including annotations, ordinal comparisons, and all intermediate results at various stages of ground truth pipeline processing
1. *scripts* - Encompasses all code used to assemble the dataset, collect annotations, generate the baseline and proposed ground truth (see figure below), and analyze and plot the results. Command line options for each script are documented in the code.


## Prerequisites
1. Anaconda with python 3.11.14 installed (we suggest using a new conda environment)
1. Once installed, install the required libraries:`pip install -r requirements.txt`
1. An R installation with `Rscript` added to the system path so it can be executed from anywhere
1. R must have the following packages installed: *data.table, irr, optparse, psych*

## Running the code
### Collecting Interval-scale Annotations with the PAGAN tool
Steps involving the PAGAN annotation tool are provided here.  Note that these steps are not necessary for users interested in applying the ordinal deformation method to their annotation data (see [the annotation fusion section below](#Annotationfusion))

**Collecting Annotations (using PAGAN):** The PAGAN annotation tool ([Melhart et al., 2019](https://ieeexplore.ieee.org/abstract/document/8925149)) was modified and used to collect annotations.  For completeness, we provide our modified PAGAN tool source and helper scripts in the `scripts/annotation/annotation_collection_and_preprocessing` folder, along with a README file explaining how to use it.  Please note that it is not necessary to use this tool for annotation collection or to run the annotation fusion code below.

**Cleaning, Aligning, Selecting (PAGAN) Annotations:** The code in `scripts/annotation/annotation_preprocessing` cleans, aligns, and selects the PAGAN annotations.  See the README file in that preprocessing folder for further instructions.  Running this code is not necessary for users interested in testing the proposed fusion method (below). At present, this code takes PAGAN annotations as input.

### Annotation Fusion
Two annotation fusion methods are provided: an *align-and-average* fusion (the paper's baseline method) and *ordinal deformation* fusion (the proposed method).  Details for running the code for each method are in `scripts/annotation/annotation_fusion/README.md`.  Note that users wanting to run the baseline or proposed methods can provide their own annotation files.
