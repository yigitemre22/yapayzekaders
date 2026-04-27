import pickle

dosyalar = {
    "Genetik":       "model_genetic.pkl",
    "YSA":           "model_ysa.pkl",
    "Decision Tree": "model_decision_trees_gini.pkl",
    "Gaussian NB":   "model_gnb.pkl",
    "Bernoulli NB":  "model_brn.pkl",
}

for name, path in dosyalar.items():
    loaded = pickle.load(open(path, "rb"))
    print(f"{name}: {type(loaded)} — keys: {loaded.keys() if isinstance(loaded, dict) else 'MODEL (dict değil!)'}")