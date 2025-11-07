from sklearn.manifold import TSNE

def run_tsne(tsne_input, perplexity=20.0, n_components=2):
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        verbose=0,
        random_state=None,
        method="barnes_hut",
        init="pca",
        metric="euclidean",
        n_jobs=-1
    ).fit_transform(tsne_input)