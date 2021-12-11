from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy import stats
import matplotlib.pyplot as plt

def score(pred, gt):
    assert len(pred) == len(gt)
    m = len(pred)

    map_ = {}
    for c in set(pred):
        map_[c] = stats.mode(gt[pred == c])[0]
    score = sum([map_[pred[i]] == gt[i] for i in range(m)])
    return score[0] / m

if __name__ == '__main__':
    iris=load_iris()
    model=KMeans(n_clusters=3)
    pred=model.fit_predict(iris.data)
    print(score(pred,iris.target))