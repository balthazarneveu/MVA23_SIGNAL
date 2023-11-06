import numpy as np
import sklearn as skl
import json
import pandas as pd
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import scipy

# DATA "UNIFORMIZATION" -> sample all signals on a regular timeline 
# Regular timeline = 10 seconds with 100 elements (we may get a bit of aliasing)
# Then we may navigate this data as a regular data structure (could fit in an excel).
# timestamps are now exactly the same for all signals.


# - la date de début de détection de l'impulsion (en ms)
# - la largeur ou durée de l'impulsion (en ms)
# - la puissance de l'impulsion (en dB / référence) 
# - l'angle theta et l'angle phi décrivant la direction dans laquelle l'impulsion est détectée (en radians)
# - la fréquence de l'impulsion (en Ghz)

def homogenize_dataset(dataset="train", data_path = Path("radars"), num_samples=100) -> pd.DataFrame:
    """This function is used to normalize and standardize a dataset.
    It converts non-uniformly sampled signals into a regular timeline
    and normalizes the values to standard units.
    The resulting DataFrame is easier to work with for further data analysis or machine learning tasks.
    Note: we may loose information due to resampling / get some aliasing.

    Parameters:
    dataset (str): The dataset to be homogenized. Default is "train".
    data_path (Path): The path to the directory containing the dataset. Default is "radars".

    Returns:
    DataFrame: A pandas DataFrame containing the homogenized dataset.
    """
    data = []
    values_units = [
        ("largeur", 1.E-3),
        ("frequence",1.E9),
        ("puissance" ,1.),("theta", 1.),
        ("phi", 1.)
    ]
    with open(data_path/f'{dataset}_labels.json') as f:
        dict_labels_current = json.load(f)
    for idx in range(799):
        pdws = np.load(data_path/ dataset / f'pdw-{idx}.npz')
        target = dict_labels_current[f'pdw-{idx}']=="nonmenace"
        dates = pdws['date']
        uniform_timestamps = np.linspace(0, 10.E3, num_samples)
        new_dic = {}
        for label, scale in values_units: #normalize to standard units (Hz, seconds)
            non_uniform_value = pdws[label]
            regular_sampling = np.interp(uniform_timestamps, xp=dates, fp=non_uniform_value)
            new_dic[label] = regular_sampling*scale
        new_dic["menace"] = not target
        diff_ts  =(dates[1:]-dates[:-1])/1000.
        new_dic["timestamps_interval_multiples"] = np.round(diff_ts/np.min(diff_ts))
        new_dic["impulsion_freq"] = 1./np.min(diff_ts)
        data.append(new_dic)
    df = pd.DataFrame.from_dict(data)
    return df


# "HANDCRAFTED" FEATURE COMPUTATION
def get_hand_crafted_features(df: pd.DataFrame, feature_dimension=None) -> Tuple[List, List]:
    inv_freq_mean = np.array([(1./el).mean() for el in df["frequence"]])
    freq_feature = np.array([(1./el).std()/((1./el).mean()) for el in df["frequence"]])
    min_pow = np.array([el.min() for el in df["puissance"]])
    largeur_mean = np.array([el.mean() for el in df["largeur"]])
    freq_std = np.array([el.std() for el in df["frequence"]])
    std_pow = np.array([el[20:50].std() for el in df["puissance"]])
    x = np.stack([freq_std, min_pow, std_pow, freq_feature, inv_freq_mean, largeur_mean], axis=1)
    if feature_dimension is not None:
        x=x[:, :feature_dimension]
    y = [1. if el else 0. for el in df["menace"]]
    return x, y



def extract_peaks(df: pd.DataFrame, add_to_df :bool=False) -> Tuple[List, List]:
    peaks_locations_list = []
    peaks_values_list = []
    for idx in range(len(df)):
        # if df_hard.menace[idx]:
        peaks = scipy.signal.find_peaks(df.puissance[idx])[0]
        peaks_locations_list.append(peaks)
        peaks_values_list.append(df.puissance[idx][peaks])
    if add_to_df:
        df["peaks_loc"] = peaks_locations_list
        df["peaks_val"] = peaks_values_list
    return peaks_locations_list, peaks_values_list


DECISION_TREE = "Decision tree"
SVM = "SVM classifier"
RANDOM_FOREST = "Random forest"
ADABOOST = "Ada Boost"
XG_BOOST = "XG Boost"
ALL_CLASSIFIERS = [DECISION_TREE, SVM, RANDOM_FOREST, ADABOOST]

def train_classifier(
        x_train, x_test, y_train, y_test,
        feature_dimension=3,
        debug=False, show=True, forced_depth=None,
        classifier=DECISION_TREE,
    ) -> Tuple[np.ndarray, int]:
    COLOR_LIST = "rgbckyp"
    if feature_dimension is None:
        x_train_shrink, x_test_shrink = x_train, x_test 
    else:
        x_train_shrink, x_test_shrink = x_train[:, :feature_dimension], x_test[:, :feature_dimension]
    accuracies = []
    
    classifiers = []
    if classifier == DECISION_TREE or classifier == RANDOM_FOREST:
        depth_list = list(range(1, 10))
        for depth in depth_list:
            if classifier == DECISION_TREE:
                dt = DecisionTreeClassifier(max_depth=depth)
            else:
                dt = RandomForestClassifier(max_depth=depth, n_estimators=10, max_features=1, random_state=73)
            dt.fit(x_train_shrink, y_train)
            classifiers.append(dt)
            y_train_pred = dt.predict(x_train_shrink)
            y_test_pred = dt.predict(x_test_shrink)
            acc_train, acc_test = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
            accuracies.append([acc_train, acc_test])
            # if debug:
            #     print(f"accuracy training {acc_train:.1f} | accuracy test {acc_test:.1f}")
    elif classifier in [SVM, ADABOOST, XG_BOOST]:
        depth_list = [1]
        best_depth = 0
        extra = {}
        if classifier == SVM:
            # clas = SVC(kernel="rbf", C=1.)
            clas = SVC(kernel="poly", degree=5, C=1.)
        elif classifier == ADABOOST:
            clas = AdaBoostClassifier(n_estimators=50)
        elif classifier == XG_BOOST:
            import xgboost as xgb
            clas = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, )
            extra = {"eval_set":[(x_test_shrink, y_test)], "verbose": 0}
        clas.fit(x_train_shrink, y_train, **extra)
        classifiers.append(clas)
        y_train_pred = clas.predict(x_train_shrink)
        y_test_pred = clas.predict(x_test_shrink)
        acc_train, acc_test = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
        accuracies.append([acc_train, acc_test])
    else:
        raise NotImplementedError(f"Classifier {classifier} not implemented")
    accuracies = np.array(accuracies)
    if debug:
        col=COLOR_LIST[(feature_dimension-1)%len(COLOR_LIST)]
        plt.plot(depth_list, accuracies[:, 0], "--"+col, alpha=0.7, label=f"accuracy train #{feature_dimension} features")
        plt.plot(depth_list, accuracies[:, 1], "-"+col, alpha=0.7, label=f"accuracy test #{feature_dimension} features")
        plt.legend()
        plt.xlabel("Depth of the decision tree")
        plt.ylabel("Classifier accuracy")
        if show:
            plt.grid()
            plt.show()
    
    best_depth = np.argmax(accuracies[:, 1]) if forced_depth is None else forced_depth
    return accuracies[best_depth, :], depth_list[best_depth], classifiers[best_depth]


# BETTER "HANDCRAFTED" FEATURE COMPUTATION
def get_better_features(df: pd.DataFrame, feature_dimension=None, whiten=False) -> Tuple[List, List]:
    # theta=np.array([np.mean(el) for el in df["theta"]]),
    # phi=np.array([np.mean(el) for el in df["phi"]]),
    # Not informative
    freq_mean = np.array([el.mean() for el in df["frequence"]])
    ligth_speed_c = 3.E8
    wave_length_lambda= ligth_speed_c/freq_mean
    power = np.array([((np.abs(el.mean())+1.E-16)) for el in df["puissance"]])
    peak_mean_width = np.array([np.mean(el[1:] - el[:-1]) for el in df["peaks_loc"]])
    peak_max_width = np.array([np.max(el[1:] - el[:-1]) for el in df["peaks_loc"]])
    peak_median_width = np.array([np.median(el[1:] - el[:-1]) for el in df["peaks_loc"]])
    feature_dict = dict(
        # freq_special = np.array([el.std() for el in df["frequence"]]) / np.array([el.mean() for el in df["frequence"]]),
        freq_feature = np.array([(1./el).std()/((1./el).mean()) for el in df["frequence"]]),
        freq_mean = freq_mean,
        freq_std = np.array([el.std() for el in df["frequence"]]),
        min_power = np.array([el.min() for el in df["puissance"]]),
        # power= power,
        # distance_target  = (power/wave_length_lambda)**(1./4.),
        distance_target = (power)**(-1/4.) * wave_length_lambda**2,
        # mean_power = np.array([((np.abs(el.mean())+1.E-12)**(1./4.)) for el in df["puissance"]]),
        # mean_power = np.array([el.mean() for el in df["puissance"]]),
        # std_power = np.array([el.std() for el in df["puissance"]]),
        number_of_peaks = np.array([len(el) for el in df["peaks_loc"]]),
        
        # impulse_freq_sq=df.impulsion_freq**2,
        peak_mean_width = peak_mean_width,
        peak_max_width = peak_max_width,
        peak_median_width = peak_median_width,
        peak_ratio = peak_max_width/peak_median_width,
        peak_vals_std = np.array([np.std(el) for el in df["peaks_val"]]),
        impulse_freq=df.impulsion_freq,
        impulse_period=1./df.impulsion_freq,
        peak_vals_mean = np.array([np.mean(el) for el in df["peaks_val"]]),
    )
    # ts_multiples = np.array([np.quantile(el, 0.9) for el in df["timestamps_interval_multiples"]])
    # x = np.stack([freq_std, min_pow, peak_len, 1./impulse_freq, peak_width, peak_vals_std, peak_vals_mean], axis=1)
    x = np.stack([el for _, el in feature_dict.items()], axis=1)
    # print(np.mean(x, axis=0))
    if whiten:
        x = (x-np.mean(x, axis=0))/np.std(x, axis=0)
        print(np.mean(x, axis=0), np.std(x, axis=0))
    if feature_dimension is not None:
        x=x[:, :feature_dimension]
    y = [1. if el else 0. for el in df["menace"]]
    return x, y, list(feature_dict.keys())

