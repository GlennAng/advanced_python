from load_dataset import load_dataset
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def convert_arxiv_category(category : str) -> str:
    """
    Convert the arxiv category by only looking at the main category. For example, from cs.AI to 'Computer Science'
    The full taxonomy can be found on https://arxiv.org/category_taxonomy
    """
    category = category.lower() # make lower case
    category = category.split(" ")[0] # if there are multiple categories (listed with spaces between them), take the first one
    category = category.split(".")[0] # split the string at the dot so that cs.ai becomes [cs, ai] and take first element which would be cs
    physics_subcategories = ["astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph", 
                             "chao-dyn", "solv-int", "patt-sol", "adap-org"]
    if category == "cs":
        return "Computer Science"
    elif category == "econ":
        return "Economics"
    elif category == "eess":
        return "Electrical Engineering and Systems Science"
    elif category in ["math", "q-alg", "alg-geom", "funct-an", "dg-ga"]:
        return "Mathematics"
    elif category in physics_subcategories:
        return "Physics"
    elif category == "q-bio":
        return "Quantitative Biology"
    elif category == "q-fin":
        return "Quantitative Finance"
    elif category == "stat":
        return "Statistics"
    else:
        return "Other"

def convert_arxiv_categories(arxiv_categories : list) -> list:
    """
    Convert the arxiv categories to their main category using list comprehension by calling the convert_arxiv_category function.
    """
    return [convert_arxiv_category(category) for category in arxiv_categories]

def create_id_category_dataframe(arxiv_ids : list, arxiv_categories : list) -> pd.DataFrame:
    """
    Create a dataframe with the arxiv ids and their categories.
    """
    df = pd.DataFrame({'arxiv_id': arxiv_ids, 'category': arxiv_categories})
    return df

def get_arxiv_distribution(df : pd.DataFrame) -> pd.Series:
    """
    Get the distribution of the arxiv categories in the dataframe.
    """
    distribution = df['category'].value_counts() # get the counts of each category
    distribution = distribution / distribution.sum() # normalize the distribution
    return distribution

if __name__ == '__main__':
    # load the two lists containing the paper ids and their arxiv categories (must have same length)
    with open('data/arxiv_ids.pkl', 'rb') as f:
        arxiv_ids = pickle.load(f)
    with open('data/arxiv_categories.pkl', 'rb') as f:
        arxiv_categories = pickle.load(f)
    assert len(arxiv_ids) == len(arxiv_categories), "arxiv_ids and arxiv_categories should have the same length"
    arxiv_categories = convert_arxiv_categories(arxiv_categories)
    df = create_id_category_dataframe(arxiv_ids, arxiv_categories)
    print(df.head())
    n_papers = df.shape[0]
    print(f"Number of papers: {n_papers}")
    arxiv_distribution = get_arxiv_distribution(df)
    print(arxiv_distribution)

    # load the kaggle dataset (already preprocessed to be much smaller so it only)
    df_kaggle = pd.read_csv('data/arxiv_categories_kaggle.csv')
    df_kaggle.columns = ['arxiv_id', 'category']
    # apply the same conversion to the kaggle dataset
    print(f"Number of papers in kaggle dataset: {df_kaggle.shape[0]}")
    df_kaggle['category'] = df_kaggle['category'].apply(convert_arxiv_category)
    df_kaggle_distribution = get_arxiv_distribution(df_kaggle)
    print(df_kaggle_distribution)

 
def plot_pie_chart(distribution: pd.Series, title: str):
    plt.figure(figsize=(8, 8))
    plt.pie(distribution, labels=distribution.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.show()

# Plot distribution of the original dataset
plot_pie_chart(arxiv_distribution, "arXiv Category Distribution (Original Dataset)")

# Plot distribution of the Kaggle dataset
plot_pie_chart(df_kaggle_distribution, "arXiv Category Distribution (Kaggle Dataset)")