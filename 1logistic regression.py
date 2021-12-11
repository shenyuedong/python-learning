### 基于逻辑回归实现乳腺癌预测
# 模型评估

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size=0.2)
model = LogisticRegression()
model.fit(X_train , y_train)
train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)

print('train score: {train_score:.6f};test score: {test_score:.6f}'.format(train_score=train_score,test_score=test_score))

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_pred=model.predict(X_test)
accuracy_score_value=accuracy_score(y_test,y_pred)
recall_score_value=recall_score(y_test,y_pred)
precision_score_value=precision_score(y_test,y_pred)
classification_report_value=classification_report(y_test,y_pred)
print("准确率：",accuracy_score_value)
print("召回率：",recall_score_value)
print("精确率：",precision_score_value)
print(classification_report_value)