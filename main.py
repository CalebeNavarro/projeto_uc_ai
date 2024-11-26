import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("./Cleaned_Students_Performance.csv")

print(data.head())

print(data.info())

for col in data.columns:
    print(f"Coluna: {col}")
    print(data[col].unique())

def categorize_performance(average_score):
    if average_score < 60:
        return "Low"
    elif 60 <= average_score < 80:
        return "Medium"
    else:
        return "High"

data['performance_category'] = data['average_score'].apply(categorize_performance)

performance_distribution = data['performance_category'].value_counts(normalize=True) * 100

print(performance_distribution)

label_encoder = LabelEncoder()
data['performance_category_encoded'] = label_encoder.fit_transform(data['performance_category'])

X = data.drop(columns=['performance_category', 'performance_category_encoded', 'total_score', 'average_score'])
y = data['performance_category_encoded']

X = pd.get_dummies(X, columns=['race_ethnicity', 'parental_level_of_education'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

def show_the_result(model, X_test, y_test, y_pred, target_names):
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))


print("Modelo 1: Árvores de Decisão")
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
show_the_result(decision_tree, X_test, y_test, y_pred_dt, label_encoder.classes_)

print("Modelo 2: Naive Bayes")
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
show_the_result(naive_bayes, X_test, y_test, y_pred_nb, label_encoder.classes_)

print("Modelo 3: Redes Neurais")
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)
show_the_result(mlp, X_test, y_test, y_pred_nn, label_encoder.classes_)
