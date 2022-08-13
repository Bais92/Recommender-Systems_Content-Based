import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


def jaccard_distance_dataframe(
        y_act: np.array,
        y_exp_dataframe: pd.DataFrame,
        include=None,
        weights=None,
        exclude=None,
):
    """
    applies the Jaccard Score to a pandas Dataframe and returns the distance for each row/column.
    This is normally useful when the similarity/distance of binary data is relevant
    :param y_act: the actual data input
    :param y_exp_dataframe: the expected data or the attributes of e.g. products, i.e. values to be fitted to.
    :param include: list of columns to use. These columns must be convertible to int
    :param weights: custom weight vector. Useful if some elements are more important than others.
    Defaults to equal weights
    :param exclude: columns to exclude
    :return: an dataframe consisting of the jaccard distance for each expected/fitted sample/attribute
    """
    weights = weights if weights is not None else np.ones(len(y_act))
    if include is not None:
        assert isinstance(include, list)
        test_df = y_exp_dataframe[include].astype(int)
        # print(weights)
        # x = [c for c in zip(test_df.columns, weights)]
        # print(x)
    elif include is None and exclude is not None:
        test_df = y_exp_dataframe.drop(exclude, axis=1).astype(int)
    else:
        test_df = y_exp_dataframe.drop(exclude, axis=1).astype(int)
    y_trans = y_act.astype(int)

    jaccard_distance = test_df.apply(
        lambda x: 1 - jaccard_score(y_trans, x, sample_weight=weights), axis=1
    )
    y_exp_dataframe["jaccard_distance"] = jaccard_distance
    y_exp_dataframe = y_exp_dataframe.sort_values(by="jaccard_distance", ascending=True)
    # Print the top 5 rows of the DataFrame
    return y_exp_dataframe


def jaccard_score_dataframe(current_item: np.array,
                            item_matrix: pd.DataFrame,
                            include=None,
                            weights: np.array = None,
                            exclude=None,
                            algorithm: str = "jaccard",
                            drop_na_input: bool = False,
                            ):
    df = item_matrix.copy()
    if include is not None:
        assert isinstance(include, list)
        df = df[include].astype(int)
    if exclude is not None:
        df = df.drop(exclude, axis=1).astype(int)
    df.loc["user input"] = current_item
    df.loc["weights"] = weights if weights else np.ones(len(df.loc["user input"]))
    if drop_na_input:
        df = df.loc[:, df.loc["user input"] == 1]
        weights = df.loc["weights"].astype(int)
    df.drop(["weights"], inplace=True)
    # Calculate all pairwise distances
    distances = pdist(df.values, metric=algorithm, w=weights)
    # Convert the distances to a square matrix
    similarity_array = 1 - squareform(distances)

    # Wrap the array in a pandas DataFrame
    similarity_df = pd.DataFrame(similarity_array, index=df.index, columns=df.index)
    return similarity_df


def euclidian_distance_dataframe():
    pass

# qs = qs.annotate(minpreis=Coalesce(Subquery(leasing_min.values("leasingrate")[:1]),F('basispreis')*2/100))

def k_nearest_neighbors_dataframe(
        x: np.array,
        dataset: pd.DataFrame,
        number_of_neighbors=3,
        include=None,
        weights=None,
        exclude=None,
):
    """
    applies the k nearest neighbor algorithm to the provided dataset
    :param x: the actual data input
    :param dataset: the expected data or the attributes of e.g. products, i.e. values to be fitted to.
    :param number_of_neighbors: Integer. The number of neighbors to be selected.
    :param include: list of columns to use. These columns must be convertible to int
    :param weights: custom weight vector. Useful if some elements are more important than others.
    Defaults to equal weights
    :param exclude: columns to exclude
    :return: an dataframe consisting of the of the k nearest neighbors in descending order.
    """
    pass
