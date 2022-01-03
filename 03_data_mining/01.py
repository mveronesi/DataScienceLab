import csv
from tqdm import tqdm
import pandas as pd
from mlxtend import frequent_patterns as fp

dataset = []

with open("../datasets/online_retail.csv") as f:
    print("READING DATABASE")
    next(f)
    for row in tqdm(csv.reader(f)):
        if row[0][0] != "C":
            dataset.append({
                "InvoiceNo": row[0],
                "StockCode": row[1],
                "Description": row[2],
                "Quantity": int(row[3]),
                "InvoiceDate": row[4],
                "UnitPrice": float(row[5]),
                "CustomerID": row[6],
                "Country": row[7]
            })
dataset.sort(key= lambda l: l["InvoiceNo"])
print("DONE")

print("AGGREGATING DATASET")
item_sets = {}
items = set()
prev = ""
for t in tqdm(dataset):
    if t["InvoiceNo"] != prev:
        prev = t["InvoiceNo"]
        item_sets[prev] = set()
    item_sets[t["InvoiceNo"]].add(t["Description"])
    items.add(t["Description"])
print("DONE")

items = list(items)

print("GENERATING MLXTEND MATRIX (which actually is a sparse matrix)")
pa_matrix = []
for set in tqdm(item_sets.values()):
    tmp = []
    for item in items:
        tmp.append(1 if item in set else 0)
    pa_matrix.append(tmp)
print("DONE")

print("GENERATING PANDAS DATAFRAME")
df = pd.DataFrame(data=pa_matrix, columns=items)
print("DONE")

print("EXTRACTING ASSOCIATION RULES")
fi = fp.fpgrowth(df, min_support=0.01, use_colnames=True)
print("Number of associaton rules", len(fi))
#print(fi.to_string())
rules = fp.association_rules(fi, metric='confidence', min_threshold=0.85)
print("DONE: found ", len(rules), " association rules")
print(rules.to_string())