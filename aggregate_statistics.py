from process_users import load_users
from scipy.stats import skew, kurtosis
import statistics

users, users_ids_to_idxs = load_users()

#Initialize total counters and lists storing individual user upvotes/downvotes
upvotes_tot = 0
downvotes_tot = 0
users_num = len(users)

# Loop through each user and sum up upvotes and downvotes
for user in users: 
    upvotes_tot += user.count_upvotes()
    downvotes_tot += user.count_downvotes()

# Compute the averages (mean)
mean_upvotes = upvotes_tot / users_num
mean_downvotes = downvotes_tot / users_num

# Print the results
print(f"Average upvotes per user: {mean_upvotes:.2f}")
print(f"Average downvotes per user: {mean_downvotes:.2f}")
        
# Median
# Create lists to hold all upvote/downvote 
# counts
upvote_counts = []
downvote_counts = []

# Loop through each user and collect their counts
for user in users:
    upvote_counts.append(user.count_upvotes())
    downvote_counts.append(user.count_downvotes())

# Calculate the median
med_upvotes = statistics.median(upvote_counts)
med_downvotes = statistics.median(downvote_counts)

# Print the results
print(f"Median upvotes per user: {med_upvotes}")
print(f"Median downvotes per user: {med_downvotes}")

# Compute skewness and kurtosis
skewness_value_up = skew(upvote_counts)
skewness_value_dn = skew(downvote_counts)
kurtosis_value_up = kurtosis(upvote_counts)
kurtosis_value_dn = kurtosis(downvote_counts)
print(f'Skewness of upvotes: {skewness_value_up:.2f}, Kurtosis of upvotes: {kurtosis_value_up:.2f}')
print(f"Skewness of downvotes: {skewness_value_dn:.2f}, Kurtosis of downvotes: {kurtosis_value_dn:.2f}")
    

# Compute the minimum and maximum values    
min_values_up = min(upvote_counts)
max_values_up = max(upvote_counts) 
min_values_dn = min(downvote_counts) 
max_values_dn = max(downvote_counts)
print(f"The minimum of upvotes among all users is: {min_values_up}, the maximum of upvotes among all users is {max_values_up}")
print(f"The minimum of downvotes among all users is: {min_values_dn}, the maximum of downvotes among all users is {max_values_dn}")
