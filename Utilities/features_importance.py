"""Module regarding features importance"""

from string import Template
from typing import List
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

def plot_features_importance(forest: RandomForestClassifier,
                             features_names: List[str],
                             top_n_features_to_plot: int = None):
    """
    Plots the features importance.

    :param forest: classifier that has already been learned.
    :param features_names: list of feature names
    :param top_n_features_to_plot: how many most important
    features should be plotted
    :rtype: None
    """
    features_names = np.array(features_names)
    if top_n_features_to_plot != None:
        features_count = top_n_features_to_plot
    else:
        features_count = len(features_names)
    features_names = features_names[:features_count]
    importances = forest.feature_importances_[:features_count]
    std = np.std([tree.feature_importances_[:features_count]
                  for tree in forest.estimators_[:features_count]],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    template = Template("$i. feature: $feature_name ($importance)")
    for i in range(features_count):
        substitutions = {"i" : i, "feature_name" : features_names[indices[i]],
                         "importance" : importances[indices[i]]}
        print(template.safe_substitute(substitutions))
    plt.figure()
    plt.title("Features importance")
    plt.bar(range(features_count), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(features_count), features_names)
    plt.xlim([-1, features_count])
    plt.show()
