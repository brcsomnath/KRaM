# Generating data

Please find the instructions below for retrieving the data for each of the datasets:


## Glove 

We provide the gender-biased word embeddings in the `glove/` folder.


## Deepmoji

Download the original data used in our experiments from [here](https://drive.google.com/file/d/1mmGHXsBYYaGIOXNHoleCcsTc5-hF26Mv/view?usp=sharing).
Unzip the contents and place the `emoji_sent_race_0.5` in the `deepmoji/` folder.

## Synthetic

To generate the synthetic data with the continous latent attribute, run the following:

```
cd synthetic/
python generate.py
```


## UCI Crimes

No additional data download needed for this data. The data is download automatically if it is not already present in the `crimes/` folder. 


## Jigsaw Toxicity Classification

To generate the data with GPT-3.5 embeddings used in our experiments, use the following steps:

1. Download the raw data (we need only `train.csv` file) from the Kaggle Challenge [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).

2. Place the file in the `jigsaw/` folder.

3. Create an OpenAI account ([link](https://platform.openai.com/)) and get an API key. 

4. Paste the key in `jigsaw/openai_api.key` file.

5. Run the following command.

```
cd jigsaw/
python gpt3.py
```

