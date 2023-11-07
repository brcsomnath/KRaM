# Kernelized Rate-Distortion Maximizer (KRaM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)


We present the implementation of the NeurIPS 2023 paper:

> [**Robust Concept Erasure via Kernelized Rate-Distortion Maximization**](),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/)<sup>1</sup>, [Nicholas Monath](https://people.cs.umass.edu/~nmonath/)<sup>2</sup>, [Avinava Dubey](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, [Amr Ahmed](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)<sup>1</sup>. <br>
UNC Chapel Hill<sup>1</sup>,  Google Deepmind<sup>2</sup>, Google Research<sup>3</sup>


## Overview
Distributed representations provide a vector space that captures meaningful relationships between data instances.  The distributed nature of these representations, however, entangles together multiple attributes or <i>concepts</i> of data instances (e.g., the topic or sentiment of a text, characteristics of the author (age, gender, etc), etc).  Recent work has proposed the task of <i>concept erasure</i>, in which rather than making a concept predictable, the goal is to remove an attribute from distributed representations while retaining other information from the original representation space as much as possible.  In this paper, we propose a new distance metric learning-based objective, the <b>K</b>ernelized <b>Ra</b>te-Distortion <b>M</b>aximizer (KRaM), for performing concept erasure. KRaM fits a transformation of representations to match a specified distance measure (defined by a labeled concept to erase) using a modified rate-distortion function. Specifically, KRaM's objective function aims to make instances with similar concept labels dissimilar in the learned representation space while retaining other information.  We find that optimizing KRaM effectively erases various types of concepts—categorical, continuous, and vector-valued variables—from data representations across diverse domains. We also provide a theoretical analysis of several properties of KRaM's objective. To assess the quality of the learned representations, we propose an alignment score to evaluate their similarity with the original representation space. Additionally, we conduct experiments to showcase KRaM's efficacy in various settings, from erasing binary gender variables in word embeddings to vector-valued variables in GPT-3 representations.

![alt text](https://github.com/brcsomnath/KRaM/blob/master/data/figures/scenarios.png?raw=true)

## Installation
The simplest way to run our code is to start with a fresh environment.
```
conda create -n KRaM python=3.6.13
source activate KRaM
pip install -r requirements.txt
```

## Data 

The detailed instructions for generating the data for all datasets is provided [here](data/README.md). 

## Running KRaM

To run KRaM on a specific dataset. Use the following command:

```
python main.py --dataset_name glove
```

The choices of dataset names are -- 'glove', 'deepmoji', 'synthetic', 'crimes', 'jigsaw_religion_openai', and 'jigsaw_gender_openai.


## Reference


```
@article{KRaM,
  title = {Robust Concept Erasure via Kernelized Rate-Distortion Maximization},
  author = {Basu Roy Chowdhury, Somnath and 
            Monath, Nicholas and
            Dubey, Kumar Avinava and
            Ahmed, Amr and
            Chaturvedi, Snigdha},
  journal={Advances in Neural Information Processing Systems},
  volume = {36},
  year = {2023}
}
```