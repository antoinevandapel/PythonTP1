import numpy as np
from PIL import Image
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier


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


YPred = []


def default_label_fn(index, original):
    return original


def pred_label_fn(index, original):
    return original + '::' + meta[YPred[index]].decode('utf-8')


def show_img(img_arr, label_arr, metadata, index, label_fn=default_label_fn):
    one_img = img_arr[index, :]
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:].reshape(32, 32)
    rgb = np.dstack([r, g, b])
    img = Image.fromarray(np.array(rgb), 'RGB')
    display(img)
    print(label_fn(index, metadata[label_arr[index][0]].decode('utf-8')))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


rel_path = "./data/"

X = unpickle(rel_path + 'data_batch_1')
img_data = X[b'data']
img_label_original = img_label = X[b'labels']
img_label = np.array(img_label).reshape(-1, 1)

test_X = unpickle(rel_path + 'test_batch')
test_data = test_X[b'data']
test_label_original = test_label = X[b'labels']
test_label = np.array(test_label).reshape(-1, 1)

sample_image_data = img_data[0:10, :]
batch = unpickle(rel_path + 'batches.meta')
meta = batch[b'label_names']

data_point_no = 10
sample_test_data = test_data[:data_point_no, :]

nbrs = CustomKNClassifier(n_neighbors=3).fit(img_data, img_label_original)
YPred = nbrs.predict(sample_test_data)

for i in range(0, len(YPred)):
    show_img(sample_test_data, test_label, meta, i, label_fn=pred_label_fn)