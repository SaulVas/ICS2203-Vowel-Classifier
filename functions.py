import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_experiment(data, k, distance_metric='minkowski', test_size=0.25, random_state=None):
    """
    Runs an experiment to classify vowel phonemes using k-NN classifier.

    Parameters:
    - data: DataFrame, the input data containing vowel phoneme samples
    - k: int, the number of nearest neighbors to consider
    - distance_metric: str, optional, the distance metric to use for
                       classification (default: 'minkowski')
    - test_size: float, optional, the proportion of the data to use for testing (default: 0.25)
    - random_state: int or None, optional, the random seed for reproducibility (default: None)

    Returns:
    - conf_matrix: array, the confusion matrix of the classification results
    - f1: float, the weighted F1 score of the classification results
    - classes: array, the unique classes in the data

    """
    female_data = data[data['Gender'] == 'F']
    male_data = data[data['Gender'] == 'M']

    # Split the data into training and test sets (75% train, 25% test)
    female_train, female_test = train_test_split(female_data,
                                                 test_size=test_size,
                                                 random_state=random_state,
                                                 stratify=female_data['Vowel Phoneme'])
    male_train, male_test = train_test_split(male_data,
                                             test_size=test_size,
                                             random_state=random_state,
                                             stratify=male_data['Vowel Phoneme'])

    train_data = pd.concat([female_train, male_train])
    test_data = pd.concat([female_test, male_test])

    x_train = train_data[['Formant 1', 'Formant 2', 'Formant 3']]
    y_train = train_data['Vowel Phoneme']
    x_test = test_data[['Formant 1', 'Formant 2', 'Formant 3']]
    y_test = test_data['Vowel Phoneme']

    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return conf_matrix, f1, knn.classes_

def analyze_k_values(data, k_values):
    """
    Analyzes the performance of the k-nearest neighbors classifier for different values of k.

    Args:
        data (numpy.ndarray): The input data for classification.
        k_values (list): A list of k values to evaluate.

    Returns:
        tuple: A tuple containing the results and the classes used for classification.
            The results dictionary contains the average confusion matrix
            and average F1 score for each k value.
            The classes list contains the unique classes present in the data.
    """
    results = {}
    for k in k_values:
        conf_matrices = []
        f1_scores = []
        for _ in range(10):
            random_state = np.random.randint(100)
            conf_matrix, f1, classes = run_experiment(data,
                                                      k,
                                                      test_size=0.25,
                                                      random_state=random_state)
            conf_matrices.append(conf_matrix)
            f1_scores.append(f1)
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        avg_f1_score = np.mean(f1_scores)
        results[k] = (avg_conf_matrix, avg_f1_score)
    return results, classes

def analyze_k_values_and_distance_metrics(data, k_values, metrics):
    results = pd.DataFrame(index=k_values, columns=metrics)
    for k in k_values:
        for metric in metrics:
            conf_matrices = []
            f1_scores = []
            for _ in range(10):
                random_state = np.random.randint(100)
                conf_matrix, f1, _ = run_experiment(data,
                                                    k,
                                                    distance_metric=metric,
                                                    test_size=0.25,
                                                    random_state=random_state)
                if conf_matrix is not None:
                    conf_matrices.append(conf_matrix)
                    f1_scores.append(f1)
            if conf_matrices:
                avg_f1_score = np.mean(f1_scores)
                results.loc[k, metric] = avg_f1_score

    # Add a row for the average F1 score per distance metric
    avg_row = results.mean(axis=0)
    avg_row.name = 'Average'
    results = pd.concat([results, avg_row.to_frame().T])

    return results

def analyze_confusion(conf_matrix, classes):
    """
    Analyzes the confusion matrix and returns a sorted list of confusion information.

    Parameters:
    conf_matrix (numpy.ndarray): The confusion matrix.
    classes (list): The list of classes.

    Returns:
    list: A sorted list of confusion information, sorted in descending order of confusion values.
    """
    confusion_info = {}
    for index1, true_class in enumerate(classes):
        for index2, predicted_class in enumerate(classes):
            if index1 != index2:
                confusion_info[(true_class, predicted_class)] = conf_matrix[index1, index2]
    sorted_confusion = sorted(confusion_info.items(), key=lambda item: item[1], reverse=True)
    return sorted_confusion
