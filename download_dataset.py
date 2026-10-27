from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download dataset 967 from the UCI Machine Learning Repository.
dataset = fetch_ucirepo(id=967)

# UCI provides the raw URL column and its target label.
urls = dataset.data.features["URL"]
uci_labels = dataset.data.targets["label"]

data = pd.DataFrame({
    "url": urls,
    # UCI: 1 = legitimate, 0 = phishing.
    # This project: "legitimate" maps to 0, "phishing" maps to 1.
    "label": uci_labels.map({1: "legitimate", 0: "phishing"}),
})

data = data.dropna(subset=["url", "label"])
data["url"] = data["url"].astype(str).str.strip()
data = data.drop_duplicates(subset=["url"])

output = Path("data")
output.mkdir(exist_ok=True)
data.to_csv(output / "urls.csv", index=False)

print(f"Saved {len(data):,} labeled URLs to data/urls.csv")
print(data["label"].value_counts())