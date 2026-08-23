
import unittest
import numpy as np

from A1 import (
    calculate_distance,
    bubble_sort,
    selection_sort,
    insertion_sort,
    get_neighbors,
    predict_class,
    weighted_predict_class,
    knn_predict,
    weighted_knn_predict
)


class TestKNNFunctions(unittest.TestCase):

    def setUp(self):

        self.X_train = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [8, 8],
            [9, 9]
        ])

        self.y_train = np.array([
            0, 0, 0, 1, 1
        ])

        self.X_test = np.array([
            [2, 2],
            [8, 8]
        ])


    def test_euclidean_distance(self):

        result = calculate_distance(
            np.array([0, 0]),
            np.array([3, 4]),
            "euclidean"
        )

        self.assertEqual(result, 5)


    def test_manhattan_distance(self):

        result = calculate_distance(
            np.array([1, 2]),
            np.array([4, 6]),
            "manhattan"
        )

        self.assertEqual(result, 7)


    def test_bubble_sort(self):

        values = [
            (5, 1),
            (2, 0),
            (4, 1),
            (1, 0)
        ]

        result = bubble_sort(values)

        distances = [x[0] for x in result]

        self.assertEqual(
            distances,
            [1, 2, 4, 5]
        )


    def test_selection_sort(self):

        values = [
            (5, 1),
            (2, 0),
            (4, 1),
            (1, 0)
        ]

        result = selection_sort(values)

        distances = [x[0] for x in result]

        self.assertEqual(
            distances,
            [1, 2, 4, 5]
        )


    def test_insertion_sort(self):

        values = [
            (5, 1),
            (2, 0),
            (4, 1),
            (1, 0)
        ]

        result = insertion_sort(values)

        distances = [x[0] for x in result]

        self.assertEqual(
            distances,
            [1, 2, 4, 5]
        )


    def test_get_neighbors(self):

        neighbors = get_neighbors(
            self.X_train,
            self.y_train,
            np.array([2, 2]),
            k=3
        )

        self.assertEqual(
            len(neighbors),
            3
        )


    def test_predict_class(self):

        neighbors = [
            (1.0, 0),
            (2.0, 0),
            (3.0, 1)
        ]

        result = predict_class(
            neighbors
        )

        self.assertEqual(result, 0)


    def test_weighted_prediction(self):

        neighbors = [
            (1.0, 0),
            (2.0, 1),
            (3.0, 1)
        ]

        result = weighted_predict_class(
            neighbors
        )

        self.assertEqual(result, 0)


    def test_knn_prediction(self):

        result = knn_predict(
            self.X_train,
            self.y_train,
            self.X_test,
            k=3
        )

        self.assertEqual(
            len(result),
            len(self.X_test)
        )


    def test_weighted_knn_prediction(self):

        result = weighted_knn_predict(
            self.X_train,
            self.y_train,
            self.X_test,
            k=3
        )

        self.assertEqual(
            len(result),
            len(self.X_test)
        )


if __name__ == "__main__":
    unittest.main()