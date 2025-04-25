from load_dataset import load_dataset
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)  # Show all rows when displaying DataFrames
import logging

# Configure logging
logging.basicConfig(filename = 'data_processing.log', filemode = 'w', format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', level = logging.INFO)
logger = logging.getLogger('paper_ratings_processor')

def transform_dates(rated_papers : pd.DataFrame) -> None:
    """
    Transform the date column to a more readable format.
    """
    n_missing_original = rated_papers["time"].isna().sum()
    logger.info(f"Number of missing Dates before filling: {n_missing_original}.")  # 8783
    # Convert from string to datetime
    # First format: without microseconds
    result = pd.to_datetime(rated_papers["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    # Identify the rows for which it failed even through the date is not missing
    mask = result.isna() & ~rated_papers["time"].isna()  # the tilde means boolean NOT (the rows where null after conversion but not before)
    # Second format: with microseconds (in those problematic cases)
    result.loc[mask] = pd.to_datetime(rated_papers.loc[mask, "time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    rated_papers["time"] = result
    n_missing_after_conversion = rated_papers["time"].isna().sum()
    logger.info(f"Number of missing Dates after conversion to date: {n_missing_after_conversion}.")  # 8783 (exactly the same as before)

def fill_missing_dates(rated_papers : pd.DataFrame, global_min_date : pd.Timestamp) -> None:
    """
    Fill the missing dates with the global min date.
    """
    # set null to the global min date (but 1 additional year to avoid being counted to the same session)
    mask = rated_papers["time"].isna()
    rated_papers.loc[mask, "time"] = global_min_date
    n_missing_after_fill = rated_papers["time"].isna().sum()
    logger.info(f"Min Date after Filling: {rated_papers['time'].min()}.")  # 2021-07-06 08:27:24 and 8783
    logger.info(f"Number of missing Dates after filling: {n_missing_after_fill}.")  # 0

def get_session_ids(rated_papers : pd.DataFrame, global_min_date : pd.Timestamp, time_threshold_seconds : int = 3600) -> None:
    """
    Group the ratings into sessions based on the time difference between consecutive ratings.
    """
    if time_threshold_seconds <= 0:
        raise ValueError("Time threshold must be positive.")
    # Sort by user_id and time
    rated_papers.sort_values(by = ["user_id", "time"], inplace = True)
    # Calculate the time difference between consecutive ratings
    time_diff = rated_papers.groupby("user_id")["time"].diff().dt.total_seconds()
    # confirm that the first value for each user is NaN
    logger.info(f"Number of NaN values in time_diff: {time_diff.isna().sum()}.")
    new_session_condition = (time_diff > time_threshold_seconds) | (time_diff.isna())
    rated_papers['session_id'] = new_session_condition.groupby(rated_papers['user_id']).cumsum()
    # Session IDs start at 1 but we set the ID of the global min date to 0
    rated_papers.loc[rated_papers["time"] == global_min_date, "session_id"] = 0
    # confirm that the number of session ID 0 is equal to the number of null date earlier
    n_zero_session_id = (rated_papers["time"] == global_min_date).sum()
    logger.info(f"Number of session_id 0: {n_zero_session_id}.")  # 8783


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


def create_sessions_df(rated_papers: pd.DataFrame) -> pd.DataFrame:
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
    logger.info(f"Sessions DataFrame Sample for User 0 without History:\n{sessions_df[sessions_df["user_id"] == 0]}")

    sessions_df['history'] = sessions_df.apply(lambda row: get_history_for_session(rated_papers, row['user_id'], row['session_id']), axis=1)
    logger.info(f"Sessions DataFrame Sample for User 0 with History:\n{sessions_df[sessions_df['user_id'] == 0]}")
    # confirm that the number of NaN history is equal to the number of users
    logger.info(f"Number of NaN history: {sessions_df['history'].isna().sum()}.")  # 8783
    # Save as CSV
    sessions_df.to_csv("data/sessions.csv", index = False)
    return sessions_df
    

if __name__ == "__main__":
    logger.info("Starting data processing...")
    # Load the dataset containing explicit user feedback
    rated_papers = load_dataset("rated_papers.csv")
    logger.info(f"Dataset sample:\n{rated_papers.head()}")
    # 4 columns arxiv_id | user_id | rating | time
    user_ids = rated_papers["user_id"].unique().tolist()  # unique user ids (duplicates in case they rated multiple papers)
    n_ratings, n_users = rated_papers.shape[0], len(user_ids)
    logger.info(f"Number of Ratings: {n_ratings}, Number of Users: {n_users}.")  # 774K ratings, 13352 users
    transform_dates(rated_papers)
    # find the min date across all users to fill the missing dates but lower it by 1 year to not be counted to the same session
    global_min_date = rated_papers["time"].min() - pd.DateOffset(years = 1)
    logger.info(f"Global Min Date across all Users lowered by 1 year: {global_min_date}.")  # 2021-07-06 08:27:24.
    fill_missing_dates(rated_papers, global_min_date)
    # Group into sessions
    get_session_ids(rated_papers, global_min_date)
    sessions_df = create_sessions_df(rated_papers)
    
    
    
#Create a class for the users with attribute of users_id #build-in function to create a class named User
        
class User:
    def __init__(self, user_id, positive_papers=None, negative_papers=None):
        # This is the unique identifier for the user
        self.user_id = user_id
        
        # These are the papers the user has upvoted (positive interaction)
        # If nothing is passed in, use an empty list
        self.positive_papers = positive_papers or []
        
        # These are the papers the user has downvoted (negative interaction)
        # Again, default to empty list if none provided
        self.negative_papers = negative_papers or []

    def count_upvotes(self):
        # Count how many positive papers the user has (i.e., number of upvotes)
        return len(self.positive_papers)

    def count_downvotes(self):
        # Count how many negative papers the user has (i.e., number of downvotes)
        return len(self.negative_papers)

    def __repr__(self):
        # This is what will print when you display the user object
        # Useful for debugging or summaries
        return f"User(user_id={self.user_id}, upvotes={self.count_upvotes()}, downvotes={self.count_downvotes()})"

   
users = []  # Create an empty list to store all user objects

# Loop through each row in the sessions_id table (your DataFrame)
for _, row in sessions_df.iterrows():
    # Create a new User object using data from the row
    user = User(
        user_id=row['user_id'],                           # ID of the user
        positive_papers=row['positive_papers'],        # List of upvoted paper IDs
        negative_papers=row['negative_papers']         # List of downvoted paper IDs
    )
    
    # Add this user to the users list
    users.append(user)    
    
    # Print the second user object (shows ID, upvotes, downvotes)
print(users[1])

# Print just their upvoted paper list
print(users[1].positive_papers)

# Print how many papers they upvoted
print(users[1].count_upvotes())

 