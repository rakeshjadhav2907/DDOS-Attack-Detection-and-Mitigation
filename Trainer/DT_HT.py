from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

class DDoSClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocess_data(self):
        data = pd.read_csv(self.data_path)
        data.iloc[:, 2] = data.iloc[:, 2].str.replace('.', '')
        data.iloc[:, 3] = data.iloc[:, 3].str.replace('.', '')
        data.iloc[:, 5] = data.iloc[:, 5].str.replace('.', '')
        X_flow = data.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')
        y_flow = data.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

    def train_model(self):
        # Define parameters for grid search
        param_grid = {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize Decision Tree classifier
        dt = DecisionTreeClassifier(random_state=0)

        # Perform grid search
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Get best parameters
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)

        # Train model with best parameters
        self.model = DecisionTreeClassifier(**best_params, random_state=0)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_flow_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_flow_pred)
        acc = accuracy_score(self.y_test, y_flow_pred)
        precision = precision_score(self.y_test, y_flow_pred)
        recall = recall_score(self.y_test, y_flow_pred)
        f1 = f1_score(self.y_test, y_flow_pred)

        print("Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

if __name__ == "__main__":
    data_path = '/content/drive/MyDrive/DDOS/FlowStatsfile.csv'
    model_path = '/content/drive/MyDrive/DDOS/models/DecisionTree_model_hyperparameter_tuned.pkl'

    ddos_classifier = DDoSClassifier(data_path)
    ddos_classifier.preprocess_data()
    ddos_classifier.train_model()
    ddos_classifier.evaluate_model()
    ddos_classifier.save_model(model_path)
