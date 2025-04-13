from load_dataset import load_dataset
import arxiv
import numpy as np
import pandas as pd
import pickle

def concatenate_datasets(rated_papers : pd.DataFrame, implicit_interactions : pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the 2 dataframes(stack them on top of each other horizontally).
    Since implicit_interactions has no rating column, we first construct it and fill with 0
    (so we have -1 for negative rating, +1 for positive rating and 0 for implicit interaction like just reading the paper).
    """
    implicit_interactions['rating'] = 0
    all_interactions = pd.concat([rated_papers, implicit_interactions], axis = 0)
    return all_interactions

def get_unique_arxiv_ids(all_interactions : pd.DataFrame) -> list:
    """
    Get the unique arxiv ids from the all_interactions dataframe (there might be duplicates if different users interact with the same paper).
    Returns approx. 350K unique arxiv ids arranged in a numpy array. Since numpy is specialized in numbers and these are strings, we transform it to a list.
    """
    unique_arxiv_ids = all_interactions['arxiv_id'].unique()
    unique_arxiv_ids = unique_arxiv_ids.tolist()
    return unique_arxiv_ids

def get_arxiv_categories(unique_arxiv_ids : list, batch_size : int = 500, max_tries : int = 10) -> list:
    """
    Get the arxiv categories for the unique arxiv ids using the arxiv API.
    Returns a list with the arxiv categories for each arxiv id.
    To avoid sending too many requests to the arxiv API, we send the requests in batches of limited size.
    """
    n_arxiv_ids = len(unique_arxiv_ids)
    arxiv_categories = []
    # delay between requests to avoid sending too many requests to the arxiv API and repeat the request if it faiils
    client = arxiv.Client(delay_seconds = 0.5)
    # iterate over the arxiv_ids but take steps of batch_size, where overshooting in the end is handled by Python
    for i in range(0, n_arxiv_ids, batch_size):
        batch = unique_arxiv_ids[i: i + batch_size]
        n_tries = 0
        # the request sometimes crash so we retry it up to max_tries times
        while n_tries < max_tries:
            try:
                search = arxiv.Search(id_list = batch)
                results = list(client.results(search))
                # append the categories to the list if request succeeds
                arxiv_categories.extend([paper.primary_category for paper in results])
                print(f"Processed samples: {i + batch_size}/{n_arxiv_ids}")
                # break the loop if the request succeeds so we don't retry it
                break
            except arxiv.HTTPError as e:
                # raise number of tries if the request fails
                n_tries += 1
                print(f"Error at samples {i}-{i + batch_size}: {e}")
        # abort the program if we reach the maximum number of tries and the request still fails
        if n_tries == max_tries:
            raise Exception(f"Failed to process samples {i}-{i + batch_size} after {max_tries} tries.")
    return arxiv_categories
    
if __name__ == '__main__':
    # load the dataframes by importing load_dataset from the provided python file (both implicit and explicit interactions)
    rated_papers = load_dataset("rated_papers.csv")
    implicit_interactions = load_dataset("implicit_interactions.csv")
    all_interactions = concatenate_datasets(rated_papers, implicit_interactions)
    unique_arxiv_ids = get_unique_arxiv_ids(all_interactions)
    with open('data/arxiv_ids.pkl', 'wb') as f:
        pickle.dump(unique_arxiv_ids, f)
    arxiv_categories = get_arxiv_categories(unique_arxiv_ids)
    with open('data/arxiv_categories.pkl', 'wb') as f:
        pickle.dump(arxiv_categories, f)