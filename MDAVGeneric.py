import secrets
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def categorical_distance(row1, row2):
    return np.sum(row1 != row2)


class MDAVGeneric:
    """
    Implements the MDAV-generic algorithm from [1].

    [1]: Domingo-Ferrer, J., & Torra, V. (2005). Ordinal, continuous and heterogeneous
    k-anonymity through microaggregation. Data Mining and Knowledge Discovery, 11, 195–212.
    """

    def __init__(self, k, feature_columns):
        self.k = k
        self.feature_columns = feature_columns
        self.clusters = None
        self.remaining = None
        self.remaining_indices = None
        self.categorical = None
        self.continuous = None
        self.distance_matrix = None

    def partition(self, df):
        self.__prepare_data(df)
        self.__build_distance_matrix()
        self.clusters = []

        while len(self.remaining_indices) >= 3 * self.k:
            centroid = self.__find_centroid()
            far_end_dist = self.__get_distance_vector_of_most_distant_point(centroid)
            self.__assign_closest_points_to_new_cluster(far_end_dist, self.k)

            other_end_dist = (
                self.__get_distance_vector_of_most_distant_point_from_data_point(
                    far_end_dist.loc[self.remaining_indices]
                )
            )
            self.__assign_closest_points_to_new_cluster(other_end_dist, self.k)

        if len(self.remaining_indices) >= 2 * self.k:
            centroid = self.__find_centroid()
            far_end_dist = self.__get_distance_vector_of_most_distant_point(centroid)
            self.__assign_closest_points_to_new_cluster(far_end_dist, self.k)

        self.clusters.append(self.remaining_indices)
        return self.clusters

    def __prepare_data(self, df):
        self.remaining = df.drop(
            columns=[
                column for column in df.columns if column not in self.feature_columns
            ]
        )
        self.remaining_indices = self.remaining.index
        self.categorical = [
            column
            for column in self.feature_columns
            if self.remaining[column].dtype == "category"
        ]
        self.continuous = [
            column
            for column in self.feature_columns
            if self.remaining[column].dtype != "category"
        ]

        if self.continuous:
            self.remaining.loc[:, self.continuous] = (
                self.remaining.loc[:, self.continuous]
                - self.remaining.loc[:, self.continuous].mean()
            ) / self.remaining.loc[:, self.continuous].std()

        for attribute in self.categorical:
            self.remaining.loc[:, attribute] = pd.factorize(
                self.remaining.loc[:, attribute]
            )[0]

    def __find_centroid(self):
        centroid_continuous = (
            self.remaining.loc[self.remaining_indices, self.continuous].mean()
            if self.continuous
            else pd.Series()
        )
        modes = (
            self.remaining.loc[self.remaining_indices, self.categorical].mode()
            if self.categorical
            else pd.DataFrame()
        )
        centroid_categorical = (
            np.array(
                [secrets.choice(modes.loc[:, column].dropna()) for column in modes]
            )
            if self.categorical
            else np.array([])
        )
        return centroid_continuous, centroid_categorical

    def __get_distance_vector_of_most_distant_point(self, position):
        pos_continuous, pos_categorical = position
        distances = np.zeros(len(self.remaining_indices))

        if self.continuous:
            distances += cdist(
                [pos_continuous.values] if not pos_continuous.empty else [[]],
                self.remaining.loc[self.remaining_indices, self.continuous].values,
                metric="euclidean",
            )[0]
        if self.categorical:
            distances += cdist(
                [pos_categorical] if len(pos_categorical) > 0 else [[]],
                self.remaining.loc[self.remaining_indices, self.categorical].values,
                metric="hamming",
            )[0] * len(self.categorical)

        max_dist_array_pos = distances.argmax()
        max_dist_index = self.remaining_indices[max_dist_array_pos]
        return self.distance_matrix.loc[max_dist_index, self.remaining_indices]

    def __get_distance_vector_of_most_distant_point_from_data_point(
        self, distance_vector
    ):
        max_dist_array_pos = distance_vector.argmax()
        max_dist_index = self.remaining_indices[max_dist_array_pos]
        return self.distance_matrix.loc[max_dist_index, self.remaining_indices]

    def __assign_closest_points_to_new_cluster(self, distance_vector, k):
        array_positions = np.argpartition(distance_vector, k)[:k]
        indices = self.remaining_indices.take(array_positions)
        self.remaining_indices = self.remaining_indices.drop(indices)
        self.clusters.append(indices)

    def __build_distance_matrix(self):
        dist_continuous = (
            pdist(self.remaining.loc[:, self.continuous].values, metric="euclidean")
            if self.continuous
            else np.zeros(len(self.remaining) * (len(self.remaining) - 1) // 2)
        )
        dist_categorical = (
            pdist(self.remaining.loc[:, self.categorical].values, metric="hamming")
            * len(self.categorical)
            if self.categorical
            else np.zeros(len(self.remaining) * (len(self.remaining) - 1) // 2)
        )
        self.distance_matrix = pd.DataFrame(
            squareform(dist_categorical + dist_continuous),
            index=self.remaining.index,
            columns=self.remaining.index,
        )
