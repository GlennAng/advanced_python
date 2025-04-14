import pandas as pd

def clean_arxiv_id(arxiv_id : str) -> str:
    """
    Clean the arxiv id by removing the .pdf suffix.
    """
    if arxiv_id.endswith(".pdf"):
        arxiv_id_after = arxiv_id[:-4]
        print(f"Cleaned arxiv id: {arxiv_id} -> {arxiv_id_after}")
        return arxiv_id_after
    else:
        return arxiv_id

def load_dataset(filename):
    # load any of our datasets from a csv file using pandas
    assert filename in ["rated_papers.csv", "implicit_interactions.csv", "abstract_sentences.csv"], 'Options for filenames are: "rated_papers.csv" or "implicit_interactions.csv" or "abstract_sentences.csv"'
    df = pd.read_csv('data/' + filename)
    # convert arxiv_id to string
    if 'arxiv_id' in df.columns:
        df['arxiv_id'] = df['arxiv_id'].astype(str)
    # Clean arxiv_id if needed
    if 'arxiv_id' in df.columns:
        df['arxiv_id'] = df['arxiv_id'].apply(clean_arxiv_id)
    return df

if __name__ == '__main__':
    print(load_dataset("rated_papers.csv").head())
    print(load_dataset("implicit_interactions.csv").head())
    print(load_dataset("abstract_sentences.csv").head())

    print(load_dataset("rated_papers.csv").shape)
    print(load_dataset("implicit_interactions.csv").shape)
    print(load_dataset("abstract_sentences.csv").shape)