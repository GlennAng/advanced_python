# Recommender Systems for scientific Documents
The folder `data` contains three files taken from https://github.com/avg-dev/scholar_inbox_datasets and described by the author below.
The two additional files `arxiv_ids.pkl` and `arxiv_categories.pkl` were constructed by calling `python generate_arxiv.py` but it takes multiple hours to run.
It collects all IDs of papers contained in either `rated_papers.csv` or `implicit_interactions.csv` and gets the corresponding arxiv category from their API.
To save time, we now simply load these 2 files as lists (both of length around 350K in which the i-th entry in `arxiv_categories.pkl` is the category of the i-th entry in `arxiv_ids.pkl`). 
On top of that, `arxiv_categories_kaggle.csv` was taken from https://www.kaggle.com/datasets/Cornell-University/arxiv and contains arxiv_id plus (potentially multiple categories).

## Recommendations
### Explicit user ratings
`rated_papers.csv`  
We publish explicit user ratings, as submitted via the thumbs up/ down buttons on papers displayed on scholar-inbox.com.  
$N=774k$
| arxiv_id | user_id | rating | time |
|---|---|---|---|

### Implicit user interactions
`implicit_interactions.csv`  
This is a dataset containing the user-paper interactions that were not explicit up/down ratings.
Such interactions include reading a paper, or clicking on any of the paper buttons to collect, share, etc.  
$N=556k$

| arxiv_id | user_id | time |
|---|---|---|


## Abstract highlighting 
`abstract_sentences.csv`  
Scholar Inbox's sentence highlighting feature was built using user-collected sentence classification data.   
We used this data to train out abstract sentence highlighting model.   
This dataset contains 2538 samples and four sentence labels: problem, task, idea, result


| arxiv_id | abstract | start_idx | end_idx | label |
|---|---|---|---|---|