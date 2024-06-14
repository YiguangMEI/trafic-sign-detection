from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import joblib
import datetime
from src.image import img
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


class model:
    def __init__(self, seed=42, n_jobs=-1):
        self.seed = seed
        self.standard_size = (100, 100)
        self.classifier = None
        self.name = None
        self.n_jobs = n_jobs

    def save(self, path="src/models"):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = path + "/model_" + date + ".pkl"
        joblib.dump(self.classifier, model_path)

    def load(self, path):
        self.classifier = joblib.load(path)

    def predict_window(self, window):
        window = img(self.standard_size, window=window)
        window.preprocess()
        return self.classifier.predict(window.data)

    def train_svm(self, train_data, max_iter=1000, verbose=0, n_jobs=-1):
        self.name = "SVM"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        self.classifier = SVC(
            kernel="linear",
            shrinking=True,
            random_state=self.seed,
            max_iter=max_iter,
            class_weight="balanced",
            verbose=verbose,
            probability=True,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        self.classifier.fit(X_train, y_train)

        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_svm2(self, train_data, max_iter=1000, verbose=0, n_jobs=-1):
        self.name = "SVM"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        svm = SVC(
            kernel="linear",
            shrinking=True,
            random_state=self.seed,
            max_iter=max_iter,
            class_weight="balanced",
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        self.classifier = BaggingClassifier(
            estimator=svm,
            n_estimators=10,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.classifier.fit(X_train, y_train)

        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_elastic_net(self, train_data, max_iter=1000, verbose=0):
        self.name = "Elastic Net"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def grid_search_svm(self, train_data, max_iter=1000, verbose=0):
        self.name = "grid linear svm.SVC"
        warnings.filterwarnings("ignore")

        # Extract data and labels from train_data
        X = np.array([img.data for img in train_data.images])
        y = np.array([img.label for img in train_data.images])

        # Define the parameter grid for grid search
        tuned_parameters = {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000],
            "class_weight": [None, "balanced"],
            "max_iter": [max_iter],
            "shrinking": [True, False],
        }

        # Initialize the SVM model
        svm = SVC()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=svm, param_grid=tuned_parameters, cv=5, verbose=verbose, n_jobs=-1
        )

        # Fit the grid search to the data
        grid_search.fit(X, y)

        # Store the best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        warnings.filterwarnings("default")

    def train_lr(self, train_data, max_iter=1000, verbose=0):
        self.name = "Logistic Regression"
        X = train_data.data
        # normalize the data
        # X = np.array(X) / 255
        y = [img.label for img in train_data.images]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            n_jobs=self.n_jobs,
        )
        print(f"Start training {self.name} with max_iter {max_iter}")
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_dual_lr(self, train_data, max_iter=1000, verbose=0):
        self.name = "dual liblinear Logistic Regression"
        X = [img.data for img in train_data.images]
        # normalize the data
        # X = np.array(X) / 255
        y = [img.label for img in train_data.images]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            dual=True,
            solver="liblinear",
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_xgboost(self, train_data, verbose=0):
        self.name = "XGBoost"
        X = [img.data for img in train_data.images]
        # normalize the data
        # X = np.array(X) / 255
        y = [img.label for img in train_data.images]
        y = train_data.LabelEncoder.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = XGBClassifier(
            n_estimators=100, random_state=self.seed, verbosity=verbose
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_OneVsRest(self, train_data, max_iter=100, verbose=0):
        self.name = "OneVsRest"
        X = [img.data for img in train_data.images]
        # normalize the data
        # X = np.array(X) / 255
        y = [img.label for img in train_data.images]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        base_clf = LogisticRegression(
            random_state=self.seed, max_iter=max_iter, verbose=verbose
        )
        # base_clf = SVC(kernel='linear', random_state=self.seed, class_weight='balanced')
        # from sklearn.ensemble import GradientBoostingClassifier
        # base_clf = GradientBoostingClassifier(n_estimators=100, random_state=self.seed,verbose=verbose)
        # base_clf = XGBClassifier(n_estimators=100, random_state=self.seed, verbosity=verbose)
        self.classifier = OneVsRestClassifier(base_clf)
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def evaluate(self, val_data):
        X = [img.data for img in val_data.images]
        y = [img.label for img in val_data.images]

        if self.name == "XGBoost" and self.label_encoder:
            y = self.label_encoder.transform(y)

        y_pred = self.classifier.predict(X)
        print(f"Accuracy on validation data: {accuracy_score(y, y_pred)}")

        # plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # Get unique labels
        unique_labels = list(set(y))

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cmap="Blues",
        )
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()

        report = classification_report(y, y_pred)
        print(report)
        return accuracy_score(y, y_pred)

    def plot_learning_curve(self, train_data, max_iter=2000, verbose=0):
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        # X = np.array(X) / 255
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # Generate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            self.classifier, X_train, y_train, verbose=verbose, cv=None
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.legend(loc="best")
        plt.show()
