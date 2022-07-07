print("FIRST LINE")
import numpy as np
import pandas as pd
from glob import glob
import os
from pathlib import Path
import time
from datetime import datetime

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.loci import LOCI
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.lmdd import LMDD
from pyod.models.pca import PCA
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from PyNomaly import loop
from ldcof import LDCOF

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc


from re_model import residual_model
#from baseline_od_model import ISO_Forest, KNN_model, Kth_NN_model, LOF_model, COF_model, LOCI_model, CBLOF_model, uCBLOF_model, LOoP_model, LDCOF_model, HBOS_model, OCSVM_model, Nu_OCSVM_model
from baseline_od_model import ISO_Forest,AutoEncoder_model, LOF_model ,KNN_model, CBLOF_model, HBOS_model,OCSVM_model
from Consolidate_baseline_Results import consolidate_result
from experiments import experiments_list, re_models

import pickle
# from sacred import Experiment
# from sacred.observers import FileStorageObserver
#ex = Experiment('ReconstructionErrorFeatures', )


# ex.observers.append(FileStorageObserver('./results/experiments'))


def baseline_Goldstein_results(filepath):
    filepath = Path(filepath)

    for file in glob(str(filepath / '*.csv')) + glob(str(filepath / '*.csv.gz')):
        print("file in baseline_Goldstein_results: ", file)
        df = pd.read_csv(file)
        print(len(df))

        ##Splitting features and label
        data = df.iloc[:, :df.shape[1] - 1]
        label = df.iloc[:, -1]
        label = np.array(label)
        label[label == 'o'] = 1
        label[label == 'n'] = 0

        features = list(df.columns)[:-1]

        num_features = len(list(df.columns)) - 1
        print("features, num_features: ", features, num_features)

        print(os.path.basename(file))
        file_basename = os.path.basename(file)
        file_basename = file_basename.split('-')



        if file_basename == ['UCI_Credit','Card.csv']:
            categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' ]
            num_features = [ 'LIMIT_BAL','AGE','BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            trainX, testX, trainy, testy = data_transform(data, label, categorical_features, num_features)

        elif file_basename == ['Adult','data.csv']:
            df = pd.read_csv(file, header=None)
            df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                          'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'net']
            ##Splitting features and label
            data = df.iloc[:, :df.shape[1] - 1]
            label = df.iloc[:, -1]
            label = np.array(label)
            label[label == ' >50K'] = 1
            label[label == ' <=50K'] = 0
            categorical_features = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
            num_features = ['age','education-num','capital-gain','capital-loss','hours-per-week']
            trainX, testX, trainy, testy = data_transform(data, label, categorical_features, num_features)
            print("trainX in adult data: ",trainX)
            print("label: ",label)

        else:
            trainX, testX, trainy, testy = train_test_split(data, label, test_size=0.3, stratify=label)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(trainX)
            trainX = scaler.transform(trainX)
            testX = scaler.transform(testX)

        contamin = np.mean(label)
        print("contamination: ", float(contamin))
        print("contamination: ", type(float(contamin)))
        ks = list(np.arange(10, 51))
        print("ks: ", ks)

        # Nearest-neighbor and Density Based Outlier Detection Algorithms
        ISO_Forest(trainX, testX, trainy, testy, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))
        AutoEncoder_model(trainX, testX, trainy, testy, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))
        KNN_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))
        # Kth_NN_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))
        LOF_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))
        #COF_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))
        # LOCI_model(trainX, testX, trainy, testy, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))

        # Clustering Based Outlier Detection Algorithm
        CBLOF_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))
        # uCBLOF_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))
        # LOoP_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))
        # LDCOF_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0]) + str('-') + str(file_basename[1]))

        # Statistical Based Outlier Detection Algorithm
        HBOS_model(trainX, testX, trainy, testy, ks, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))

        # Classifier Based Outlier Detection Algorithm
        OCSVM_model(trainX, testX, trainy, testy, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))
        # Nu_OCSVM_model(trainX, testX, trainy, testy, float(contamin), str(file_basename[0])+str('-')+str(file_basename[1]))

        print('\n\n')

    print("All Baseline OD model training is completed")
    print("Gathering all results and find the best parameters")
    # consolidate_result()
    print("Completed result considation and storing best parameteres")

def data_transform(data, label, categorical_features, num_features):
    # standardizing and one hot encoding data
    categorical_features_transform = pd.get_dummies(data[categorical_features])
    data_onehot_encoded = pd.concat([categorical_features_transform, data[num_features]], axis=1)

    trainX, testX, trainy, testy = train_test_split(data_onehot_encoded, label, test_size=0.3, stratify=label)

    # instatiating scalers
    scaler = MinMaxScaler(feature_range=(0, 1))
    # ohe = OneHotEncoder(sparse=False)
    # Scale and Encode separately
    num_transformed_trainX = scaler.fit_transform(trainX[num_features])
    num_transformed_testX = scaler.transform(testX[num_features])

    categorical_trainX = trainX.drop(num_features, axis=1, inplace=False)
    categorical_testX = testX.drop(num_features, axis=1, inplace=False)
    trainX_new = np.concatenate([categorical_trainX, num_transformed_trainX], axis=1)
    testX_new = np.concatenate([categorical_testX, num_transformed_testX], axis=1)
    return trainX_new, testX_new, trainy, testy

def eval_metrics(y_true, y_prediction, outlierness_score):
    model_precision, model_recall, _ = precision_recall_curve(np.array(y_true), np.array(outlierness_score),
                                                              pos_label=1)
    model_f1 = f1_score(y_true, y_prediction)
    model_auc = auc(model_recall, model_precision)
    model_accuracy = accuracy_score(y_true, y_prediction)
    model_roc_auc = roc_auc_score(y_true, outlierness_score)
    model_fpr, model_tpr, _ = roc_curve(y_true, outlierness_score)

    return model_accuracy, model_f1, model_auc, model_roc_auc


def eval_metrics1(y_true, outlierness_score):
    model_precision, model_recall, _ = precision_recall_curve(np.array(y_true), np.array(outlierness_score),
                                                              pos_label=1)
    model_auc = auc(model_recall, model_precision)
    model_roc_auc = roc_auc_score(y_true, outlierness_score)
    model_fpr, model_tpr, _ = roc_curve(y_true, outlierness_score)

    return 'NA', 'NA', model_auc, model_roc_auc


def train_data_evaluation(trainX, trainy, clf, start_time, od_algo, input_data, i, j, results_roc_train,
                          results_pr_auc_train):
    # print("results_roc_train: ", results_roc_train.shape)
    # print("results_pr_auc_train: ", results_pr_auc_train.shape)
    # print("i, j: ", i, j)
    time_taken = round((time.time() - start_time), 4)
    y_true = trainy
    y_true = y_true.astype(np.int64)

    if od_algo == 'LDCOF':
        outlierness_score = clf.predict_proba(trainX)
        model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics1(y_true, outlierness_score)
        results_roc_train[i][j], results_pr_auc_train[i][j] = model_roc_auc, model_auc
        print(
            "Training Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %s %.s %.4f %.4f %.4f " % (
            input_data, od_algo, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))
        return results_roc_train, results_pr_auc_train

    y_prediction = clf.predict(trainX)
    outlierness_score = clf.predict_proba(trainX)
    outlierness_score = outlierness_score[:, 1]
    model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics(y_true, y_prediction, outlierness_score)
    results_roc_train[i][j], results_pr_auc_train[i][j] = model_roc_auc, model_auc
    print(
        "Training Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %.4f %.4f %.4f %.4f %.4f " % (
        input_data, od_algo, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))
    return results_roc_train, results_pr_auc_train


def test_data_evaluation(testX, testy, clf, od_algo, input_data, i, j, results_roc_test, results_pr_auc_test):
    # print("results_roc_test: ", results_roc_test.shape)
    # print("results_pr_auc_test: ", results_pr_auc_test.shape)
    # print("i, j: ", i, j)
    start_time = time.time()
    y_true = testy
    y_true = y_true.astype(np.int64)

    if od_algo == 'LDCOF':
        outlierness_score = clf.predict_proba(testX)
        time_taken = round((time.time() - start_time), 4)
        model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics1(y_true, outlierness_score)
        results_roc_test[i][j], results_pr_auc_test[i][j] = model_roc_auc, model_auc
        print(
            "Test Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %s %s %.4f %.4f %.4f " % (
            input_data, od_algo, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))
        return results_roc_test, results_pr_auc_test

    y_prediction = clf.predict(testX)
    time_taken = round((time.time() - start_time), 4)
    outlierness_score = clf.predict_proba(testX)
    outlierness_score = outlierness_score[:, 1]
    model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics(y_true, y_prediction, outlierness_score)
    results_roc_test[i][j], results_pr_auc_test[i][j] = model_roc_auc, model_auc
    print(
        "Test Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %.4f %.4f %.4f %.4f %.4f " % (
        input_data, od_algo, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))
    return results_roc_test, results_pr_auc_test


def re_od_model_result(trainX, testX, trainy, testy, od_model, algo_name, input_data, i, j, results_roc_train,
                       results_pr_auc_train, results_roc_test, results_pr_auc_test):
    start_time = time.time()
    clf = od_model
    clf.fit(trainX)
    results_roc_train, results_pr_auc_train = train_data_evaluation(trainX, trainy, clf, start_time, algo_name,
                                                                    input_data, i, j, results_roc_train,
                                                                    results_pr_auc_train)
    results_roc_test, results_pr_auc_test = test_data_evaluation(testX, testy, clf, algo_name, input_data, i, j,
                                                                 results_roc_test, results_pr_auc_test)
    return results_roc_train, results_pr_auc_train, results_roc_test, results_pr_auc_test


def LOop_Result(trainX, testX, trainy, testy, neighbor, algo_name, input_data, i, j, results_roc_train,
                results_pr_auc_train, results_roc_test, results_pr_auc_test):
    start_time = time.time()
    clf = loop.LocalOutlierProbability(trainX, extent=1, n_neighbors=neighbor, use_numba=True).fit()
    time_taken = round((time.time() - start_time), 4)
    y_true = trainy
    y_true = y_true.astype(np.int64)
    outlierness_score = clf.local_outlier_probabilities
    model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics1(y_true, outlierness_score)
    results_roc_train[i][j], results_pr_auc_train[i][j] = model_roc_auc, model_auc
    print(
        "Training Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %s %s %.4f %.4f %.4f " % (
        input_data, algo_name, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))

    start_time = time.time()
    clf = loop.LocalOutlierProbability(testX, extent=1, n_neighbors=neighbor, use_numba=True).fit()
    time_taken = round((time.time() - start_time), 4)
    y_true = testy
    y_true = y_true.astype(np.int64)
    outlierness_score = clf.local_outlier_probabilities
    model_accuracy, model_f1, model_auc, model_roc_auc = eval_metrics1(y_true, outlierness_score)
    results_roc_test[i][j], results_pr_auc_test[i][j] = model_roc_auc, model_auc
    print(
        "Training Data (%s), %s Results (model_accuracy, model_f1, model_auc, model_roc_auc, Training-time): %s %s %.4f %.4f %.4f " % (
        input_data, algo_name, model_accuracy, model_f1, model_auc, model_roc_auc, time_taken))
    return results_roc_train, results_pr_auc_train, results_roc_test, results_pr_auc_test


def save_output(output_directory, dataframe_list):
    for i, j in dataframe_list:
        csv_file = str(output_directory) + '/' + str(j) + '.csv'
        i.to_csv(csv_file)


# @ex.config  # collects local vars as configs
def my_config():
    if os.path.exists('data'):
        filepath = 'data'
    else:
        # filepath = r'R:\Zentrale\ZB-S\Daten\S5\S52\S52-1\Daten\dfki_re'
        filepath = 'C:/Users/nimee/MLprojs/Outlier detection/DataSets'

    training_option = 1  # 0: Baseline Goldstein paper results, 1: Our Approach- Results using RE features
    # re parameters
    experiments_list = experiments_list
    start_i = 0
    end_i = len(experiments_list)
    re_models = re_models
    # seed = 28
    # re_model = GradientBoostingRegressor(n_estimators=100, subsample=.5, max_depth=3, random_state=seed)
    # acc_range = [0.05, 0.99]  # [0.05, 0.99] or []
    # clip = False
    # metric = 'mse'  # 'mse', 'mae', 'err'
    # scale = 'minmax'  # 'standard', 'minmax'


# @ex.automain  # parses cmd args and provides vars
def main(filepath, training_option, experiments_list, start_i, end_i, re_models, _run):
    if training_option == 0:
        baseline_Goldstein_results(filepath)  # For Baseline Goldstein paper results
    else:  # For Our Approach: Results using RE features
        filepath = Path(filepath)
        tabs = []
        ##print('Experiment:', _run._id, '\n\n')
        files = glob(str(filepath / '*.csv')) + glob(str(filepath / '*.csv.gz'))
        files = [file for file in files if Path(file).name not in ['kdd99-unsupervised-ad.csv']]

        print("start_i, end_i: ", start_i, end_i)
        for exp in experiments_list[start_i:end_i]:
            start_time1 = time.time()
            re_model, acc_range, clip, metric, scale, seed = exp
            if len(acc_range) == 0:
                output_directory = './results/re_approach_results/Exp_' + str(
                    experiments_list.index(exp)) + '_re_model_' + str(re_model) + '_acc_range_[]_clip_' + str(
                    clip) + '_metric_' + str(metric) + '_scale_' + str(scale) + '_seed_' + str(seed)
            else:
                output_directory = './results/re_approach_results/Exp_' + str(
                    experiments_list.index(exp)) + '_re_model_' + str(re_model) + '_acc_range_' + str(
                    acc_range[0]) + '_to_' + str(acc_range[1]) + '_clip_' + str(clip) + '_metric_' + str(
                    metric) + '_scale_' + str(scale) + '_seed_' + str(seed)
            if not Path(output_directory).exists():
                Path(output_directory).mkdir(parents=True)
            print("output directory created: ", output_directory)

            # cols = [os.path.basename(name_file) for name_file in files]
            #            cols = ['pen-local-unsupervised-ad.csv', 'pen-global-unsupervised-ad.csv', 'breast-cancer-unsupervised-ad.csv', 'speech-unsupervised-ad.csv', 'aloi-unsupervised-ad.csv', 'shuttle-unsupervised-ad.csv', 'letter-unsupervised-ad.csv', 'satellite-unsupervised-ad.csv', 'annthyroid-unsupervised-ad.csv']
            cols = ['breast-cancer-unsupervised-ad.csv']
            #            od_algo_list = ['ISO', 'KNN', 'KthNN', 'LOF', 'CBLOF', 'uCBLOF', 'LOoP', 'LDCOF', 'HBOS', 'OCSVM', 'Nu-OCSVM']
            od_algo_list = ['ISO', 'HBOS']
            # datasets = ['pen-local-unsupervised-ad.csv', 'pen-global-unsupervised-ad.csv', 'breast-cancer-unsupervised-ad.csv', 'speech-unsupervised-ad.csv', 'aloi-unsupervised-ad.csv', 'shuttle-unsupervised-ad.csv', 'letter-unsupervised-ad.csv', 'satellite-unsupervised-ad.csv', 'annthyroid-unsupervised-ad.csv']
            datasets = ['pen-global-unsupervised-ad.csv']
            results_roc_combined_train = np.zeros((len(od_algo_list), len(datasets)))
            results_pr_auc_combined_train = np.zeros((len(od_algo_list), len(datasets)))
            results_roc_combined_test = np.zeros((len(od_algo_list), len(datasets)))
            results_pr_auc_combined_test = np.zeros((len(od_algo_list), len(datasets)))
            results_roc_only_re_train = np.zeros((len(od_algo_list), len(datasets)))
            results_pr_auc_only_re_train = np.zeros((len(od_algo_list), len(datasets)))
            results_roc_only_re_test = np.zeros((len(od_algo_list), len(datasets)))
            results_pr_auc_only_re_test = np.zeros((len(od_algo_list), len(datasets)))
            cols.insert(0, 'OD_Algo')
            print("cols: ", cols)
            df_roc_combined_train, df_roc_combined_test, df_roc_onlyre_train, df_roc_onlyre_test = pd.DataFrame(
                columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
            df_prauc_combined_train, df_prauc_combined_test, df_prauc_onlyre_train, df_prauc_onlyre_test = pd.DataFrame(
                columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

            # files = files[:2]
            for file in files:
                print('dataset:', file)
                df = pd.read_csv(file)
                print('number of observations:', len(df))

                data = df.iloc[:, :df.shape[1] - 1]
                label = df.iloc[:, -1]
                label = np.array(label)
                label[label == 'o'] = 1
                label[label == 'n'] = 0

                features = list(df.columns)[:-1]
                num_features = len(list(df.columns)) - 1
                file_basename = os.path.basename(file)
                # file_basename = file_basename.split('-')
                contamin = np.mean(label)
                print("contamination: ", float(contamin))

                trainX, testX, trainy, testy = train_test_split(data, label, test_size=0.3, stratify=label,
                                                                random_state=seed)
                print("number of features:", num_features)
                print("trainX, testX, trainy, testy: ", trainX.shape, testX.shape, trainy.shape, testy.shape)

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(trainX)
                trainX = scaler.transform(trainX)
                testX = scaler.transform(testX)

                model = re_models[re_model]
                X_re_features, X_re_train, X_re_test, train_scores, valid_scores, models = residual_model(trainX, testX,
                                                                                                          features,
                                                                                                          num_features,
                                                                                                          model,
                                                                                                          acc_range,
                                                                                                          clip, metric,
                                                                                                          scale)

                Combined_features = X_re_features + features
                Combined_X_train = np.column_stack((X_re_train, trainX))
                Combined_X_test = np.column_stack((X_re_test, testX))

                best_parameters = pd.read_csv('best_parameter_value.csv')

                od_re_models = {'ISO': IForest(n_estimators=int(
                    best_parameters[file_basename].values[list(best_parameters['OD_Algo'].values).index('ISO')]),
                                               max_samples="auto", contamination=contamin, max_features=1.,
                                               behaviour='old', random_state=28),
                                'KNN': KNN(contamination=contamin, n_neighbors=int(
                                    best_parameters[file_basename].values[
                                        list(best_parameters['OD_Algo'].values).index('KNN')]), method='mean'),
                                'KthNN': KNN(contamination=contamin, n_neighbors=int(
                                    best_parameters[file_basename].values[
                                        list(best_parameters['OD_Algo'].values).index('KthNN')]), method='largest'),
                                'LOF': LOF(n_neighbors=int(best_parameters[file_basename].values[
                                                               list(best_parameters['OD_Algo'].values).index('LOF')]),
                                           algorithm='auto', leaf_size=30),
                                # 'COF': COF(contamination=contamin, n_neighbors=int(best_parameters[file_basename].values[list(best_parameters['OD_Algo'].values).index('COF')])),
                                # 'LOCI': LOCI(contamination=contamin, alpha=0.5, k=3),
                                'CBLOF': CBLOF(n_clusters=int(best_parameters[file_basename].values[
                                                                  list(best_parameters['OD_Algo'].values).index(
                                                                      'CBLOF')]), contamination=contamin,
                                               use_weights=True, random_state=28),
                                'uCBLOF': CBLOF(n_clusters=int(best_parameters[file_basename].values[
                                                                   list(best_parameters['OD_Algo'].values).index(
                                                                       'uCBLOF')]), contamination=contamin,
                                                use_weights=False, random_state=28),
                                'LOoP': loop,
                                'LDCOF': LDCOF(n_clusters=int(best_parameters[file_basename].values[
                                                                  list(best_parameters['OD_Algo'].values).index(
                                                                      'LDCOF')])),
                                'HBOS': HBOS(n_bins=int(best_parameters[file_basename].values[
                                                            list(best_parameters['OD_Algo'].values).index('HBOS')]),
                                             contamination=contamin),
                                'OCSVM': OCSVM(nu=best_parameters[file_basename].values[
                                    list(best_parameters['OD_Algo'].values).index('OCSVM')], gamma=0.9,
                                               cache_size=20000, contamination=contamin),
                                'Nu-OCSVM': OCSVM(gamma=best_parameters[file_basename].values[
                                    list(best_parameters['OD_Algo'].values).index('Nu-OCSVM')], cache_size=20000,
                                                  contamination=contamin)
                                }

                for name in od_re_models.keys():
                    if name == 'LOoP':
                        results_roc_combined_train, results_pr_auc_combined_train, results_roc_combined_test, results_pr_auc_combined_test = LOop_Result(
                            Combined_X_train, Combined_X_test, trainy, testy, int(best_parameters[file_basename].values[
                                                                                      list(best_parameters[
                                                                                               'OD_Algo'].values).index(
                                                                                          'LOoP')]), name,
                            'Combined Data', od_algo_list.index(name), datasets.index(file_basename),
                            results_roc_combined_train, results_pr_auc_combined_train, results_roc_combined_test,
                            results_pr_auc_combined_test)
                        if X_re_train.shape[
                            1] > 0: results_roc_only_re_train, results_pr_auc_only_re_train, results_roc_only_re_test, results_pr_auc_only_re_test = LOop_Result(
                            X_re_train, X_re_test, trainy, testy, int(best_parameters[file_basename].values[
                                                                          list(best_parameters['OD_Algo'].values).index(
                                                                              'LOoP')]), name, 'Only Rec-Err Feature',
                            od_algo_list.index(name), datasets.index(file_basename), results_roc_only_re_train,
                            results_pr_auc_only_re_train, results_roc_only_re_test, results_pr_auc_only_re_test)
                        print('\n')
                        continue
                    results_roc_combined_train, results_pr_auc_combined_train, results_roc_combined_test, results_pr_auc_combined_test = re_od_model_result(
                        Combined_X_train, Combined_X_test, trainy, testy, od_re_models[name], name, 'Combined Feature',
                        od_algo_list.index(name), datasets.index(file_basename), results_roc_combined_train,
                        results_pr_auc_combined_train, results_roc_combined_test, results_pr_auc_combined_test)
                    if X_re_train.shape[
                        1] > 0: results_roc_only_re_train, results_pr_auc_only_re_train, results_roc_only_re_test, results_pr_auc_only_re_test = re_od_model_result(
                        X_re_train, X_re_test, trainy, testy, od_re_models[name], name, 'Only Rec-Err Feature',
                        od_algo_list.index(name), datasets.index(file_basename), results_roc_only_re_train,
                        results_pr_auc_only_re_train, results_roc_only_re_test, results_pr_auc_only_re_test)
                    print('\n')
                print('\n\n')

            for i in range(results_roc_combined_train.shape[0]):
                df_roc_combined_train.loc[i] = [od_algo_list[i]] + list(results_roc_combined_train[i])
                df_roc_combined_test.loc[i] = [od_algo_list[i]] + list(results_roc_combined_test[i])
                df_prauc_combined_train.loc[i] = [od_algo_list[i]] + list(results_pr_auc_combined_train[i])
                df_prauc_combined_test.loc[i] = [od_algo_list[i]] + list(results_pr_auc_combined_test[i])
                df_roc_onlyre_train.loc[i] = [od_algo_list[i]] + list(results_roc_only_re_train[i])
                df_roc_onlyre_test.loc[i] = [od_algo_list[i]] + list(results_roc_only_re_test[i])
                df_prauc_onlyre_train.loc[i] = [od_algo_list[i]] + list(results_pr_auc_only_re_train[i])
                df_prauc_onlyre_test.loc[i] = [od_algo_list[i]] + list(results_pr_auc_only_re_test[i])

            dataframe_list = [(df_roc_combined_train, 'ROC_for_Combined_feature_in_train'),
                              (df_roc_combined_test, 'ROC_for_Combined_feature_in_test'),
                              (df_prauc_combined_train, 'PR-AUC_for_Combined_feature_in_train'),
                              (df_prauc_combined_test, 'PR-AUC_for_Combined_feature_in_test'),
                              (df_roc_onlyre_train, 'ROC_for_Only-Re_feature_in_train'),
                              (df_roc_onlyre_test, 'ROC_for_Only-Re_feature_in_test'),
                              (df_prauc_onlyre_train, 'PR-AUC_for_Only-Re_feature_in_train'),
                              (df_prauc_onlyre_test, 'PR-AUC_for_Only-Re_feature_in_test')]
            save_output(output_directory, dataframe_list)
            print("Total time taken for one experiment combination: ", round((time.time() - start_time1), 4))
            print('\n\n\n')

filepath = './datasets'
training_option = 0
start_i =1
print("JOB STARTED")
main(filepath, training_option, experiments_list, 1, 1, re_models, 0)
#main(filepath, training_option, experiments_list, start_i, end_i, re_models, _run)