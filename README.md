# Recommender Systems for scientific Documents

Install the conda environment via the file `environment.yml` if necessary (but they are just basic libraries).

### Group Members:
Glenn Angrabeit - 5681972,
Niclas Linder - 5424597,
Aspasia Charalampopoulou - 7016190

## 1. Dataset Description
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

## 2. Data Processing
Our goal was to transform the data from `rated_papers.csv` so that user ratings are grouped into sessions.
We say that two ratings belong to the same session if their interactions took place within 1 hour.
This produces a table of the following form:

| user_id | session_id | time | positive_papers | negative_papers | history
|---|---|---|---|---|---|

- **user_id:** The ID of the user who voted on at least one paper during this session.
- **session_id:** The ID of the session displayed in this row. Counting starts at 1 for each user (so we might have user_id : 0, session_id : 1 but also user_id : 9, session_id : 1).
- **time:** The time at which the session started (the minimum of all rating times in that session).
- **positive_papers:** A string containing all arXiv IDs of papers that the user upvoted during this session (ordered by time and separated by white spaces). If there were only downvotes during this session, the entry is Nan:
- **negative_papers:** Analogously for the downvoted papers during this session.
- **history:** A string containing all arXiv IDs of papers that the user voted on before this session (ordered by time and separated by white spaces). In order to display the type of explicit feedback, each ID is appended with either "+1" for an upvote or "-1" for a downvote.

#### THIS DATA FILE IS QUITE LARGE (200 MB) SO YOU MUST FIRST CONSTRUCT IT YOURSELF:
Run `python process_users.py` to create `data/cached_sessions_df.pkl` and `data/cached_first_interactions.pkl` (might take a minute). You can then also look at `user_processing.log` to see an example for the user with ID 0.

## 3. Data Post-Processing
There are 3 files you can run that all use the data generated in steps 1 and 2.
### 3.1 Printing User Info
Run `python print_user.py` to get information on one specific user (the terminal will prompt you to enter any user ID).
### 3.2 Aggregate Statistics
Run `python aggregate_statistics.py` to get aggregate statistics with respect to the user votes.
### 3.3 Parallelization Experiment
Run `python parallelization_experiment.py` to compute the days since the first interaction for each user (sequentially AND in parallel using joblib).
We perform this experiment twice: once just like one would normally and once calling time.sleep(0.0001) to simulate complex processing.
Without sleeping, using joblibs actually causes an overhead and slows down the program. But when including the sleeping, parallelization speeds up the process from 20s to 2s.
The file further includes unit tests to make sure that the sequential and parallel results are identical.









