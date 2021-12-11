# 基于k-近邻算法实现鸢尾花分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
if __name__=='__main__':
    iris=load_iris()
    x_train,x_test,y_train,y_test=train_test_split(iris.data[:,[1,3]],iris.target)
    model=KNN()
    model.fit(x_train,y_train)

train_score=model.score(x_train,y_train)
test_score=model.score(x_test,y_test)
print("train score:",train_score)
print("test score:",test_score)