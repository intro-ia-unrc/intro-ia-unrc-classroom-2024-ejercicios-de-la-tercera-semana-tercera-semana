from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
    
data, target = load_iris(return_X_y = True) # target has numeric values 0, 1, 2 for setosa, versicolor and virginica, resp.

train_data, test_data, train_target, test_target = train_test_split(data, target) 

model = MLPClassifier(solver = 'sgd', max_iter = 100, hidden_layer_sizes = (10, 10))
model.fit(train_data, train_target)

prediction = model.predict(test_data)

print('Accuracy: ', accuracy_score(prediction,test_target))

confussion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
# TODO: fill in confussion_matrix

for line in confussion_matrix:
    print(line)





