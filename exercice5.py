import numpy as np

class CustomKNClassifier:
    def __init__(self, n_neighbors=8):
        self.data = None
        self.output = None
        self.n_neighbors = n_neighbors

    def fit(self, data, output):
        self.data = data
        self.output = output
        return self

    def predict(self, data):
        predictions = []
        for a in data:
            distance_array = []
            for index, b in enumerate(self.data):
                distance_array.append({
                    "index": index,
                    "distance": np.linalg.norm(a - b)
                })
            distance_array = sorted(distance_array, key=lambda d: d['distance'])
            distance_array = distance_array[:self.n_neighbors]
            predictions.append(self.mode(distance_array))
        return predictions

    def mode(self, distance_array):
        frequency_list = {}
        for data in distance_array:
            key = self.output[data['index']]
            if key in frequency_list:
                frequency_list[key] += 1
            else:
                frequency_list[key] = 1
        return max(frequency_list, key=frequency_list.get)