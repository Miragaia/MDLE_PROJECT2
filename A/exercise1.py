from pyspark import SparkContext

# Spark Initialization
sc = SparkContext(appName="PeopleYouMightKnow")

# Load input file into an RDD (each line is one user and their friends with tab separation)
# Format: user_id \t friend1,friend2,...
# Example: 1\t2,3,4
lines = sc.textFile("soc-LiveJournal1Adj.txt")

# --- STEP 1: Parse input into (user, set of friends) format ---
def safe_parse(line):
    """
    Parses each line into (user, set(friends)).
    Ignores lines without friends or badly formatted lines.
    """
    parts = line.strip().split("\t")
    if len(parts) != 2 or not parts[1].strip():
        return None
    try:
        user = int(parts[0])
        friends = set(map(int, parts[1].split(",")))
        return (user, friends)
    except:
        return None

# Clean and parse valid lines
user_friends = lines.map(safe_parse).filter(lambda x: x is not None)

# --- STEP 2: Broadcast friend mappings for fast lookup later ---
user_friend_map = user_friends.collectAsMap()  # Convert to dictionary
user_friend_bcast = sc.broadcast(user_friend_map)  # Share with all nodes

# --- STEP 3: Generate (friend_pair, 1) for mutual friend counting ---
def generate_candidate_pairs(user, friends):
    """
    For each user, emit pairs of friends that share that user as a mutual friend.
    Only emits pairs (A, B) where A < B to avoid duplication.
    """
    for friend1 in friends:
        for friend2 in friends:
            if friend1 < friend2:
                yield ((friend1, friend2), 1)

# Compute counts of mutual friends between user pairs
mutual_counts = user_friends.flatMap(lambda x: generate_candidate_pairs(x[0], x[1])) \
                            .reduceByKey(lambda a, b: a + b)

# --- STEP 4: Convert mutual friend counts into user-level recommendations ---
recommendations = mutual_counts.flatMap(lambda x: [
    (x[0][0], (x[0][1], x[1])),  # A → B
    (x[0][1], (x[0][0], x[1]))   # B → A
])

# --- STEP 5: Remove direct friends and self-connections ---
def filter_direct(user, recs):
    """
    Filters out users that are already friends with the given user,
    or the user themselves.
    """
    direct_friends = user_friend_bcast.value.get(user, set())
    return [(other, count) for (other, count) in recs
            if other not in direct_friends and other != user]

# --- STEP 6: Keep top 10 recommendations per user ---
top_recommendations = recommendations.groupByKey() \
    .map(lambda x: (
        x[0],
        sorted(
            filter_direct(user=x[0], recs=list(x[1])),
            key=lambda r: (-r[1], r[0])  # Sort by mutual friend count desc, then user ID asc
        )[:10]  # Limit to top 10
    )) \
    .map(lambda x: f"{x[0]}\t{','.join(str(r[0]) for r in x[1])}")

# --- STEP 7: Save results to a single text file ---
results = top_recommendations.collect()
with open("recommendations.txt", "w") as f:
    for line in results:
        f.write(line + "\n")

# Stop the Spark context
sc.stop()
