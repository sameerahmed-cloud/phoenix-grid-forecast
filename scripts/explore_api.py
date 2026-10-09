"""One-off script to inspect the EIA region-data route. Not part of the pipeline."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("EIA_API_KEY")
ROUTE = "https://api.eia.gov/v2/electricity/rto/region-data"

# 1. Route metadata: frequencies, data columns, facets, and the available date range
meta = requests.get(ROUTE, params={"api_key": KEY}, timeout=30).json()["response"]
print("Frequencies:", [f["id"] for f in meta["frequency"]])
print("Data columns:", list(meta["data"].keys()))
print("Facets:", [f["id"] for f in meta["facets"]])
print("Available range:", meta.get("startPeriod"), "to", meta.get("endPeriod"))

# 2. All allowed values for a facet. Confirms respondent codes and their names.
for facet in ["respondent", "type"]:
    values = requests.get(f"{ROUTE}/facet/{facet}", params={"api_key": KEY}, timeout=30).json()
    for v in values["response"]["facets"]:
        if facet == "type" or v["id"] in ("AZPS", "SRP") or "Arizona" in v["name"]:
            print(facet, "->", v["id"], "=", v["name"])