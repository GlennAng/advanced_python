from collections import defaultdict
from load_dataset import load_dataset
import logging
import pandas as pd
import os
pd.set_option('display.max_rows', None)  # Show all rows when displaying DataFrames

#Create a class for the users with attribute of users_id #build-in function to create a class named User
class User:
    def __init__(self, user_id, positive_papers=None, negative_papers=None, first_interaction=None):
        # This is the unique identifier for the user
        self.user_id = user_id
        
        # These are the papers the user has upvoted (positive interaction)
        # If nothing is passed in, use an empty list (the OR operator returns the first truthy value or the last value if none are truthy)
        self.positive_papers = positive_papers or []
        
        # These are the papers the user has downvoted (negative interaction)
        # Again, default to empty list if none provided
        self.negative_papers = negative_papers or []

        # The date when the user interacted the first time
        self.first_interaction = first_interaction or []

    def count_upvotes(self):
        # Count how many positive papers the user has (i.e., number of upvotes)
        return len(self.positive_papers)

    def count_downvotes(self):
        # Count how many negative papers the user has (i.e., number of downvotes)
        return len(self.negative_papers)

    def __repr__(self):
        # This is what will print when you display the user object
        # Useful for debugging or summaries
        return (f"User(user_id={self.user_id}, upvotes={self.count_upvotes()}, "
                f"downvotes={self.count_downvotes()}, first_interaction={self.first_interaction})")

def transform_dates(rated_papers : pd.DataFrame, logger) -> pd.DataFrame:
    """
    Transform the date column to a more readable format.
    """
    # count how many dates are missing before we have done any processing
    n_missing_original = rated_papers["time"].isna().sum()
    logger.info(f"Number of missing Dates before filling: {n_missing_original}.")  # 8783

    # Convert from string to datetime (first format: without microseconds)
    # errors = "coerce" means that if the conversion fails, the value will be set to NaT (Not a Time)
    result = pd.to_datetime(rated_papers["time"], format = "%Y-%m-%d %H:%M:%S", errors = "coerce")
    # The previous step fails for dates in the wrong format => they are set to NaT (Not a Time) even though the date is not missing in the original dataset
    # => we need to identify those rows (where the ampersand means boolean AND while tilde means boolean NOT)
    mask = result.isna() & ~rated_papers["time"].isna()
    # Log the number of such entries
    logger.info(f"Number of Dates in wrong format without microseconds: {mask.sum()}.")

    # Convert from string to datetime (second format for the problematic cases: with microseconds)
    # We use the mask to select only the problematic rows and convert them to datetime
    result.loc[mask] = pd.to_datetime(rated_papers.loc[mask, "time"], format = "%Y-%m-%d %H:%M:%S.%f", errors = "coerce")
    rated_papers["time"] = result
    n_missing_after_conversion = rated_papers["time"].isna().sum()
    logger.info(f"Number of missing Dates after conversion to date: {n_missing_after_conversion}.")  # 8783 (exactly the same as before)
    assert n_missing_after_conversion == n_missing_original, "The number of missing dates should not change after conversion to datetime."
    return rated_papers

def fill_missing_dates(rated_papers : pd.DataFrame, global_min_date : pd.Timestamp, logger) -> pd.DataFrame:
    """
    Fill the missing dates with the global min date.
    """
    # set null to the global min date (but 1 additional year to avoid being counted to the same session)
    mask = rated_papers["time"].isna()
    rated_papers.loc[mask, "time"] = global_min_date
    n_missing_after_fill = rated_papers["time"].isna().sum()
    assert n_missing_after_fill == 0, "There should be no missing dates after filling."
    logger.info(f"Number of missing Dates after filling: {n_missing_after_fill}.")  # 0
    logger.info(f"Min Date after Filling: {rated_papers['time'].min()}.")  # 2021-07-06 08:27:24 and 8783
    return rated_papers

def get_session_ids(rated_papers : pd.DataFrame, global_min_date : pd.Timestamp, time_threshold_seconds : int, logger) -> pd.DataFrame:
    """
    Group the ratings into sessions based on the time difference between consecutive ratings.
    """
    if time_threshold_seconds <= 0:
        raise ValueError("Time threshold must be positive.")
    # Sort by user_id and time
    rated_papers.sort_values(by = ["user_id", "time"], inplace = True)
    # Calculate the time difference between consecutive ratings
    time_diff = rated_papers.groupby("user_id")["time"].diff().dt.total_seconds()
    # Confirm that the first value for each user is NaN
    logger.info(f"Number of NaN values in time_diff: {time_diff.isna().sum()}.")
    # Create a new session if the time difference is greater than the threshold or if it's NaN
    new_session_condition = (time_diff > time_threshold_seconds) | (time_diff.isna())
    rated_papers['session_id'] = new_session_condition.groupby(rated_papers['user_id']).cumsum()
    # Session IDs for all users start at 1 but we set the ID of the global min date to 0 (helps to remember that these were originally null)
    rated_papers.loc[rated_papers["time"] == global_min_date, "session_id"] = 0
    # Confirm that the number of session ID 0 is equal to the number of null date earlier
    n_zero_session_id = (rated_papers["time"] == global_min_date).sum()
    logger.info(f"Number of session_id 0: {n_zero_session_id}.")  # 8783
    return rated_papers

def get_history_for_session(rated_papers : pd.DataFrame, user_id : int, session_id : int) -> str:
    """
    Given the user_id and session_id, returns the IDs of all papers the user has rated in previous sessions.
    The IDs are returned as a string, with each ID followed by its rating (+1 or -1).
    """
    # Get all papers from previous sessions for this user
    previous_papers = rated_papers[(rated_papers['user_id'] == user_id) & (rated_papers['session_id'] < session_id)]
    # Create history string with paper IDs and ratings
    if previous_papers.empty:
        return float('nan')  # No previous papers, return NaN
    previous_papers["rating_str"] = previous_papers["rating"].apply(lambda x: "+1" if x == 1 else "-1")
    history_items = previous_papers['arxiv_id'] + previous_papers['rating_str']
    # Join with white spaces
    history = " ".join(history_items)
    return history

def create_sessions_df(rated_papers: pd.DataFrame, logger) -> pd.DataFrame:
    """Create a DataFrame similar to the MIND dataset"""
    # Create a new DataFrame with the columns user_id, session_id and the beginning timestamp of the session (the minimum time)
    sessions_df = rated_papers.groupby(['user_id', 'session_id']).agg({'time': 'min'}).reset_index()
    # Create a new column with a string of all the positive arxiv_ids in that session (separated by whitespaces). If no positive papers, gives NaN
    positive_papers_column = rated_papers[rated_papers['rating'] == 1].groupby(['user_id', 'session_id'])['arxiv_id'].apply(lambda x: ' '.join(x))
    positive_papers_column = positive_papers_column.rename('positive_papers').reset_index()
    positive_papers_column['positive_papers'] = positive_papers_column['positive_papers']
    sessions_df = sessions_df.merge(positive_papers_column, on = ['user_id', 'session_id'], how = 'left')
    # Do the exact same for the negative papers
    negative_papers_column = rated_papers[rated_papers['rating'] == -1].groupby(['user_id', 'session_id'])['arxiv_id'].apply(lambda x: ' '.join(x))
    negative_papers_column = negative_papers_column.rename('negative_papers').reset_index()
    negative_papers_column['negative_papers'] = negative_papers_column['negative_papers']
    sessions_df = sessions_df.merge(negative_papers_column, on = ['user_id', 'session_id'], how = 'left')

    logger.info(f"Rated Papers DataFrame Sample for User 0:\n{rated_papers[rated_papers['user_id'] == 0]}")
    logger.info(f"Sessions DataFrame Sample for User 0 without History:\n{sessions_df[sessions_df['user_id'] == 0]}")

    sessions_df['history'] = sessions_df.apply(lambda row: get_history_for_session(rated_papers, row['user_id'], row['session_id']), axis=1)
    logger.info(f"Sessions DataFrame Sample for User 0 with History:\n{sessions_df[sessions_df['user_id'] == 0]}")
    # Confirm that the number of NaN history is equal to the number of users
    logger.info(f"Number of NaN history: {sessions_df['history'].isna().sum()}.")  # 8783
    return sessions_df

def process_sessions(rated_papers : pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Processing sessions...")
    # get a list of all user_ids without duplicates
    user_ids = rated_papers["user_id"].unique().tolist()
    n_ratings, n_users = rated_papers.shape[0], len(user_ids)
    logger.info(f"Number of Ratings: {n_ratings}, Number of Users: {n_users}.")

    rated_papers = transform_dates(rated_papers, logger)
    global_min_date = rated_papers["time"].min() - pd.DateOffset(years = 1)
    logger.info(f"Global Min Date across all Users lowered by 1 year: {global_min_date}.")

    rated_papers = fill_missing_dates(rated_papers, global_min_date, logger)
    rated_papers = get_session_ids(rated_papers, global_min_date, 3600, logger)

    sessions_df = create_sessions_df(rated_papers, logger)
    sessions_df.to_pickle("data/cached_sessions_df.pkl")
    logger.info("Cached sessions_df saved.")

def load_sessions_df() -> pd.DataFrame:
    if os.path.exists("data/cached_sessions_df.pkl"):
        sessions_df = pd.read_pickle("data/cached_sessions_df.pkl")
    else:
        if os.path.exists("cached_sessions_df.pkl"):
            sessions_df = pd.read_pickle("cached_sessions_df.pkl")
        else:
            raise FileNotFoundError("No cached sessions_df found. Please run the process_users.py file first.")
    return sessions_df

def process_first_interactions(sessions_df : pd.DataFrame, rated_papers : pd.DataFrame, logger) -> dict:
    logger.info("Processing first interactions...")
    # Ensure 'time' is properly converted to datetime before calculating first interactions
    rated_papers["time"] = pd.to_datetime(rated_papers["time"], errors="coerce")
    # Drop rows where 'time' could not be converted (e.g., NaT values)
    rated_papers = rated_papers.dropna(subset=["time"])
    first_interactions = rated_papers.groupby('user_id')['time'].min().to_dict()
    # Cache first_interactions
    pd.to_pickle(first_interactions, "data/cached_first_interactions.pkl")
    logger.info("Cached first_interactions saved.")
    return first_interactions

def load_first_interactions() -> dict:
    if os.path.exists("data/cached_first_interactions.pkl"):
        first_interactions = pd.read_pickle("data/cached_first_interactions.pkl")
    else:
        if os.path.exists("cached_first_interactions.pkl"):
            first_interactions = pd.read_pickle("cached_first_interactions.pkl")
        else:
            raise FileNotFoundError("No cached first_interactions found. Please run the process_users.py file first.")
    return first_interactions

def process_users(sessions_df : pd.DataFrame, first_interactions : dict) -> tuple:
    users = []  # Create an empty list to store all user objects
    # Use a defaultdict to store user data with default values for positive and negative papers (no need to check if they exist)
    user_dict = defaultdict(lambda: {"positive": [], "negative": []})
    for _, row in sessions_df.iterrows():
        user_id = row['user_id']
        # Check if the positive_papers column is a string before splitting
        if isinstance(row['positive_papers'], str):
            user_dict[user_id]["positive"].extend(row['positive_papers'].split())
        # Check if the negative_papers column is a string before splitting
        if isinstance(row['negative_papers'], str):
            user_dict[user_id]["negative"].extend(row['negative_papers'].split())
    # Creating user objects
    for user_id, papers in user_dict.items():
        user = User(
            user_id=user_id,
            positive_papers=papers["positive"],
            negative_papers=papers["negative"],
            first_interaction=first_interactions.get(user_id)
        )
        users.append(user)
    # Get a dictionary mapping user IDs to their indices in the users list
    users_ids_to_idxs = {user.user_id: idx for idx, user in enumerate(users)}
    return users, users_ids_to_idxs

def load_users() -> tuple:
    sessions_df = load_sessions_df()
    first_interactions = load_first_interactions()
    users, users_ids_to_idxs = process_users(sessions_df, first_interactions)
    return users, users_ids_to_idxs
    
if __name__ == "__main__":
    logging.basicConfig(filename = "user_processing.log", filemode = "w", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", level = logging.INFO)
    logger = logging.getLogger("paper_ratings_processor")
    logger.info("Starting user processing...")

    rated_papers = load_dataset("rated_papers.csv")
    logger.info(f"Loaded dataset with {rated_papers.shape[0]} rows and {rated_papers.shape[1]} columns.")
    logger.info(f"Dataset sample:\n{rated_papers.head()}")

    sessions_df = process_sessions(rated_papers, logger)
    first_interactions = process_first_interactions(sessions_df, rated_papers, logger)
    logger.info("User processing completed.")