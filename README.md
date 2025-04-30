# Recommender Systems for scientific Documents

## Dataset Description
Our project is based on [Scholar Inbox](https://scholar-inbox.com), a recommender system for scientific papers run here at the University of Tübingen. 
Their dataset was made publicly available under https://github.com/avg-dev/scholar_inbox_datasets a few weeks ago. You can find these three files in the folder `data` of our repository, but we mainly worked with `rated_papers.csv` and `implicit_interactions.csv` (described below).

### Explicit User Ratings 
On Scholar Inbox, users can register to receive recommendations of recently published scientific papers that might align with their interests. In order to tune these recommendation, they mave provide explicit feedback in the form of upvotes and downvotes. The file `rated_papers.csv` collects this data in the following form ($N=774K$):
| arxiv_id | user_id | rating | time |
|---|---|---|---|
- **arxiv_id:** The ID of the paper that was voted on.
- **user_id:** The ID of the user who performed the voting.
- **rating:** +1 for upvote, -1 for downvote.
- **time:** Timestamp of when the voting was done. The system did not track this data right away so many of the older entries have a null-value here.

### Implicit User Interactions
Very similar to the previous file but `implicit_interactions.csv` missing the column **rating** as it refers to implicit feedback: interactions like reading a paper or sharing it without explicitly voting on it ($N=556K$):
| arxiv_id | user_id | time |
|---|---|---|

### arXiv Paper Categories
The first thing we wanted to do is obtain the [arXiv Category](https://arxiv.org/category_taxonomy) for each scientific paper that appears in either `rated_papers.csv` or `implicit_interactions.csv`. To accomplish this, we used the python library `arxiv` and queried their API. The results are stored in the two files `data/arxiv_ids.pkl` and `data/arxiv_categories.pkl`.
- `data/arxiv_ids.pkl`: Returns a list of strings with the arxiv paper IDs, i.e. ['2004.12909', '2004.13664', ...]
- `data/arxiv_categories.pkl`: Returns a list of strings with the arxiv paper categories, i.e. ['cs.LG', 'cs.LG', ...]

You can generate these files yourself by running `python generate_arxiv.py` but the API is quite unresponsive so it took multiple hours. Instead, we just directly load these two files now. Both lists have $N = 350K$ and the idea is that the i-th entry of the first list contains the ID of the paper whose arXiv category is stored in the i-th entry of the second list.


### Comparing Scholar Inbox Paper Categories to Kaggle Paper Categories
Outside of Scholar Inbox, there is a larger ($N = 2.5M$) arXiv dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). As Scholar Inbox is mainly used by computer scientists, we wanted to see whether their distribution across paper categories is different from the general Kaggle dataset. Because this dataset is quite large, we removed everything except the two columns **arxiv_id** and **categories** from it. You do not have to download anything from the website yourself, as it is stored in `data/arxiv_categories_kaggle.csv`. 

You can generate the visualizations `plots/piecharts.pdf` and `plots/mirrorchart.pdf` by running `python create_arxiv_piecharts.py`. They show a much larger portion of Computer Science and Electrical Engineering papers for Scholar Inbox, whereas Kaggle contains way more Physics and Mathematics papers.







