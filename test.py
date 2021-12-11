import torch

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X=cancer.data
y=cancer.target

print('data shape:{0};positive:{1},negative:{2}'.format(X.shape,y[y==1].shape,y[y==0].shape))
print('腺癌数据的前两行为：')
print(cancer.data[0:2])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
print('train score:{0};test score{1}'.format(train_score,test_score))
