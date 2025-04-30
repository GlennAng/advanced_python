from process_users import load_users

users, users_ids_to_idxs = load_users()
while True:
    try:
        # Getting user input
        u = int(input(f"Enter one of the {len(users)} User IDs. For example one out of: [0, 1, 3, 6, 9]: "))
        # Check if input is valid
        if u in users_ids_to_idxs:

            # Show positive papers
            print(f"Positive papers of the user: {users[users_ids_to_idxs[u]].positive_papers}")

            # Show negative papers
            print(f"Negative papers of the user: {users[users_ids_to_idxs[u]].negative_papers}")

            # Number of positive upvotes
            print(f"Number of papers the user upvoted: {users[users_ids_to_idxs[u]].count_upvotes()}")

            # Number of negative upvotes
            print(f"Number of papers the user downvoted: {users[users_ids_to_idxs[u]].count_downvotes()}")

            # First date of interaction
            print(f"User {u} first interacted on: {users[users_ids_to_idxs[u]].first_interaction}")
                
            # Break if everything worked
            break
        else:
            print("Please enter a valid ID within the range.")
    except ValueError:
        print("Please enter a valid integer for the user ID.")