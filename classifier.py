
from pycaret.classification import *
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd
import pathlib

base_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(base_dir)
base_cwd = os.getcwd()

import warnings
warnings.filterwarnings('always')


def change_cwd_data(customer):
    new_cwd = os.path.join(base_cwd, "customer_data")
    os.chdir(new_cwd)

def change_cwd_file(customer):
    new_cwd = os.path.join(base_cwd, "customer_reports", customer)
    os.chdir(new_cwd)

def change_cwd_model(customer):
    new_cwd = os.path.join(base_cwd, "customer_models")
    os.chdir(new_cwd)

def fetch_training_data(customer):
    training_data = pd.read_csv(customer + "_data.csv",encoding= "ISO-8859-1")
    print("training_data shape",training_data.shape)
    return training_data

def sanity_check(training_data,sum_col,priority_col):
    training_data = training_data[[sum_col,priority_col]]
    training_data = training_data.drop_duplicates()
    print("training_data_shape_after_duplicates_removal", training_data.shape)
    print(training_data.isna().sum())
    training_data = training_data.dropna()
    print("training_data_shape_after_null_removal", training_data.shape)
    print(training_data.dtypes)
    return training_data

def split_target(training_data,sum_col,priority_col):
    print("Training data:",training_data.shape)
    value_counts = training_data[priority_col].value_counts()
    print("value_counts:",value_counts)
    value_counts = pd.DataFrame(value_counts)
    value_counts = value_counts.reset_index()
    value_counts.columns = ['feature','count']
    print("value_counts_before_required_classes:",value_counts)
    required_feature = list(value_counts[value_counts['count']>50]['feature'])
    print("required_feature",required_feature)
    training_data = training_data[training_data[priority_col].isin(required_feature)]
    #print(training_data)
    training_data = training_data[[sum_col,priority_col]]
    print("value_counts_after_required_classes:", training_data[priority_col].value_counts())
    #print(training_data.head())
    return training_data

def train_multiple_models(training_data,priority_col):
    cls = setup(data= training_data,
                target= priority_col,
                train_size=0.80,
                fix_imbalance=True,
                #normalize = True,
                html=False)
    #best_model = compare_models(include=['lr', 'dt', 'knn', 'rf'])
    #best_model = compare_models()
    best_model = compare_models(include=['dt','rf'])
    return best_model

def tune_best_model(best_model):
    final_best = finalize_model(best_model)
    #final_best = tune_model(final_best, n_iter=10, choose_better=True)
    metrics_frame = pull()
    metrics_frame.to_csv('final_model_metrics.csv')
    return final_best

def model_visualization_metrics(final_best):
    plot_model(final_best, plot='class_report', save=True)
    print("plot_class_report")
    plot_model(final_best, plot='confusion_matrix', save=True)
    print("plot_confusion_matrix")
    plot_model(final_best, plot='error', save=True)
    print("plot_error")
    plot_model(final_best, plot='learning', save=True)
    print("plot_learning_curve")
    plot_model(final_best, plot='auc', save=True)
    print("plot_roc_auc")
    plot_model(final_best, plot='learning', save=True)
    print("plot_feature_importance")

def finalizing_model(final_best):
    final_model = finalize_model(final_best)
    print(final_model)
    return final_model

def save_best_model(customer,final_model):
    save_best_model = save_model(final_model, customer+'_new_trained_model')
    print(save_best_model)

def initiate_process(customer,sum_col,priority_col):
    change_cwd_data(customer)
    training_data = fetch_training_data(customer)
    # starting time to train
    start_time = time.time()
    training_data = sanity_check(training_data,sum_col,priority_col)
    training_data = split_target(training_data,sum_col,priority_col)
    best_model = train_multiple_models(training_data,priority_col)
    change_cwd_file(customer)
    final_best = tune_best_model(best_model)
    # elapsed time to train
    end_time = time.time() - start_time
    change_cwd_file(customer)
    model_visualization_metrics(final_best)
    change_cwd_model(customer)
    final_model = finalizing_model(final_best)
    change_cwd_model(customer)
    save_best_model(customer,final_model)

    # measuring the total loading time to fetch data in hours, minutes and seconds format
    total_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    results = {"training completed for the customer": customer,
               "total time to train the model": total_time +" seconds",
               "the total no.of Rows": training_data.shape[0],
               "the total no.of columns": training_data.shape[1]
                }
    return results


def detect_assigned_group(customer,short_description):
    change_cwd_model(customer)
    print("-------",os.getcwd())
    loaded_model = load_model(customer+'_new_trained_model')
    new_doc = pd.DataFrame({"short_description":[short_description]})
    predict_assigned = predict_model(loaded_model, data = new_doc, encoded_labels = False)
    predict_assign = pd.DataFrame(predict_assigned)
    print(predict_assign)
    results_lst = []
    for index, row in predict_assign.iterrows():
        results_lst.append({'short_description': str(short_description), 'assigned_group': row['prediction_label']})

    results = {"Results":results_lst}
    print(results)
    return results
