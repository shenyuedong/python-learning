# sklearn决策树算法类库内部实现是使用了调优过的CART树算法，既可以做分类(DecisionTreeClassifier)，又可以做回归(DecisionTreeRegressor)，两者参数几乎一致，部分意义有差别。
#
# C4.5为优化过的ID3算法，改进：
#
#     1：用信息增益率来选择属性，克服了用信息增益选择属性时偏向选择取值多的属性不足；
#     2：在树构造过程中进行剪枝；
#     3：能够完成对连续属性的离散化处理；
#     4：能够对不完整数据进行处理。

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
if __name__=="__main__":
    wine=load_wine();
    x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target)
    clf=DecisionTreeClassifier(criterion="entropy")
    clf.fit(x_train,y_train)

train_score=clf.score(x_train,y_train)
test_score=clf.score(x_test,y_test)
print("train score:",train_score)
print("test score:",test_score)


