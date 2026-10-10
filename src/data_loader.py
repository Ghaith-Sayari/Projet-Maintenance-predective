import pandas as pd
from ucimlrepo import fetch_ucirepo

# Configuration du cas métier (le reste du code reste générique)
DATASETS = {
    "AI4I 2020 – Maintenance prédictive (UCI)": {
        "uci_id": 601,
        "target": "Machine failure",
        "positive": 1,
        "exclude": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
    }
}

def load_uci(uci_id: int) -> pd.DataFrame:
    d = fetch_ucirepo(id=uci_id)
    return pd.concat([d.data.features, d.data.targets], axis=1)