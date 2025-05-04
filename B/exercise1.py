from pyspark import SparkContext
import csv

# Spark Initialization
sc = SparkContext(appName="AssociationRules")

# --- Step 1: Read and Parse the Dataset ---

def parse_csv(line):
    """Parse a CSV line into fields"""
    try:
        return next(csv.reader([line]))
    except:
        return None

# Read CSV file
lines = sc.textFile("conditions.csv")
header = lines.first()
data = lines.filter(lambda x: x != header).map(parse_csv).filter(lambda x: x is not None)

# Build (PATIENT, set(CODE)) baskets
patient_conditions = data.map(lambda x: (x[2], x[4])) \
                         .groupByKey() \
                         .mapValues(lambda codes: set(codes))

# Count total patients
total_patients = patient_conditions.count()

# --- Step 2: Count Supports for Items, Pairs, and Triples ---

# Single item counts
item_counts = patient_conditions.flatMap(lambda x: [(item, 1) for item in x[1]]) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .collectAsMap()

# 2-itemsets counts
def generate_2_itemsets(conditions):
    """Generate all unique 2-item combinations"""
    items = list(conditions)
    return [(tuple(sorted([items[i], items[j]])), 1) for i in range(len(items)) for j in range(i+1, len(items))]

pair_counts = patient_conditions.flatMap(lambda x: generate_2_itemsets(x[1])) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .collectAsMap()

# 3-itemsets counts
def generate_3_itemsets(conditions):
    """Generate all unique 3-item combinations"""
    items = list(conditions)
    return [(tuple(sorted([items[i], items[j], items[k]])), 1)
            for i in range(len(items))
            for j in range(i+1, len(items))
            for k in range(j+1, len(items))]

triple_counts = patient_conditions.flatMap(lambda x: generate_3_itemsets(x[1])) \
                                   .reduceByKey(lambda a, b: a + b) \
                                   .collectAsMap()

# --- Step 3: Sanity Checks ---

# Support threshold
support_threshold = 1000

# Get frequent itemsets
frequent_items = [item for item, count in item_counts.items() if count >= support_threshold]
frequent_pairs = [(pair, count) for pair, count in pair_counts.items() if count >= support_threshold]
frequent_triples = [(triple, count) for triple, count in triple_counts.items() if count >= support_threshold]

print("\n--- SANITY CHECKS ---")
print("Count of frequent 1-itemsets:", len(frequent_items))
print("Count of frequent 2-itemsets:", len(frequent_pairs))
print("Count of frequent 3-itemsets:", len(frequent_triples))

# Find 3rd most frequent pair
top_3_pairs = sorted(frequent_pairs, key=lambda x: -x[1])
if len(top_3_pairs) >= 3:
    print("3rd most frequent pair:", top_3_pairs[2][0])

# Find 4th most frequent triplet
top_4_triples = sorted(frequent_triples, key=lambda x: -x[1])
if len(top_4_triples) >= 4:
    print("4th most frequent triplet:", top_4_triples[3][0])
print("---------------------\n")

# --- Step 4: Generate Association Rules ---

rules = []

# Generate (X) -> Y rules
for (X, Y), suppXY in pair_counts.items():
    if suppXY < support_threshold:
        continue

    suppX = item_counts.get(X, 1)
    suppY = item_counts.get(Y, 1)

    confidence = suppXY / suppX
    support_X = suppX / total_patients
    support_Y = suppY / total_patients
    support_XY = suppXY / total_patients

    lift = confidence / support_Y
    interest = support_XY - (support_X * support_Y)
    denominator = (max(support_X, support_Y) / (support_X * support_Y)) - 1
    standardized_lift = (lift - 1) / denominator if denominator != 0 else 0

    if standardized_lift >= 0.2:
        rules.append(((X,), Y, standardized_lift, lift, confidence, interest))

# Generate (X,Y) -> Z rules
for (X, Y, Z), suppXYZ in triple_counts.items():
    if suppXYZ < support_threshold:
        continue

    suppXY = pair_counts.get((X, Y), 1)

    confidence = suppXYZ / suppXY
    support_XY = suppXY / total_patients
    support_Z = item_counts.get(Z, 1) / total_patients
    support_XYZ = suppXYZ / total_patients

    lift = confidence / support_Z
    interest = support_XYZ - (support_XY * support_Z)
    denominator = (max(support_XY, support_Z) / (support_XY * support_Z)) - 1
    standardized_lift = (lift - 1) / denominator if denominator != 0 else 0

    if standardized_lift >= 0.2:
        rules.append(((X, Y), Z, standardized_lift, lift, confidence, interest))

# --- Step 5: Save Rules ---

# Sort rules by standardized lift descending
rules_sorted = sorted(rules, key=lambda x: -x[2])

with open("association_rules.txt", "w") as f:
    for antecedent, consequent, s_lift, lift, conf, interest in rules_sorted:
        antecedent_str = ",".join(antecedent)
        f.write(f"{antecedent_str} -> {consequent} | Standardized Lift: {s_lift:.4f} | Lift: {lift:.4f} | Confidence: {conf:.4f} | Interest: {interest:.6f}\n")

# --- Step 6: Stop Spark ---
sc.stop()
