import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions import (analyze_k_values,
                       analyze_k_values_and_distance_metrics,
                       run_experiment,
                       analyze_confusion)

if __name__ == '__main__':
    data = pd.read_csv('vowels.csv')

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    distance_metrics = ['minkowski', 'euclidean', 'manhattan']

    # Q1
    k_results, classes = analyze_k_values(data, k_values)
    for k, (conf_matrix, avg_f1_score) in k_results.items():
        print(f"K={k}: Average F1 Score: {avg_f1_score:.2f}")

    # Q2
    results = analyze_k_values_and_distance_metrics(data, k_values, distance_metrics)
    print(results)

    # Q4
    conf_matrix, _, classes = run_experiment(data, k=5)
    sorted_confusion = analyze_confusion(conf_matrix, classes)
    print("Top 5 most confused phoneme pairs:")
    for (true_class, predicted_class), count in sorted_confusion[:5]:
        print(f"{true_class} -> {predicted_class}: {count} times")

    # Plot the confusion matrix for K=5 as an example
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for K=5')
    plt.savefig('confusion_matrix_k5.png')
    plt.close()
