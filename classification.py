#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:14:12 2019
@author: hananhindy

This script has been updated to run on modern versions of scikit-learn
by replacing deprecated functions, fixing data type issues, and
ADDING FUNCTIONALITY TO SAVE TRAINED MODELS with joblib.
"""
import pandas as pd
import numpy as np
import os
import argparse
import joblib # <-- 1. IMPORT JOBLIB

# --- FIX: Import ColumnTransformer ---
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# --- END FIX ---

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

# Helper Function
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# --- FIX: Rename global variable ---
preprocessor = None
# --- END FIX ---

def load_file(path, mode, is_attack = 1, label = 1, folder_name='Bi/', sliceno = 0, verbose = True):
    # --- FIX: Use global preprocessor ---
    global preprocessor
    # --- END FIX ---
    
    columns_to_drop_packet = ['timestamp', 'src_ip', 'dst_ip']
    columns_to_drop_uni = ['proto', 'ip_src', 'ip_dst']
    columns_to_drop_bi = ['proto', 'ip_src', 'ip_dst']
    
    if os.path.getsize(path)//10 ** 9 > 0:
        x = np.zeros((0,0))
        # --- FIX: Added low_memory=False ---
        for chunk in pd.read_csv(path, chunksize=10 ** 6, low_memory=False):
            
            # --- START FIX: Clean data inside chunk ---
            chunk = chunk.fillna(-1)
            if 'is_attack' in chunk.columns:
                chunk['is_attack'] = pd.to_numeric(chunk['is_attack'], errors='coerce').fillna(-1).astype(int)
            # --- END FIX ---

            if mode == 0:
                chunk.drop(columns=columns_to_drop_packet, inplace=True, errors='ignore')
                chunk = chunk[chunk.columns.drop(list(chunk.filter(regex='mqtt')))]
            
            with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
                f.write('{}, {} \n'.format(path, chunk.shape[0]))   
                
            # --- FIX: Added .copy() ---
            x_temp = chunk.loc[chunk['is_attack'] == is_attack].copy()   
            x_temp.drop('is_attack', axis = 1, inplace = True)

            # --- START NEW FIX: Force all feature columns (except protocol) to numeric ---
            if mode == 0:
                numeric_cols_chunk = x_temp.columns[1:]
                for col in numeric_cols_chunk:
                    x_temp[col] = pd.to_numeric(x_temp[col], errors='coerce').fillna(-1)
            # --- END NEW FIX ---
            
            # --- FIX: This block is for large files, needs same ColumnTransformer logic ---
            if preprocessor == None:
                numeric_features = list(range(1, x_temp.shape[1]))
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('protocol_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [0]) # [0] is the 'protocol' column
                    ],
                    remainder='passthrough' # Pass through all other columns
                )
                x_processed = preprocessor.fit_transform(x_temp)
            else:
                x_processed = preprocessor.transform(x_temp)
            
            if hasattr(x_processed, "toarray"):
                x_temp = x_processed.toarray()
            else:
                x_temp = x_processed
            # --- END FIX ---
            
            # --- FIX: Removed problematic np.unique call ---
            # x_temp = np.unique(x_temp, axis = 0) 
            # --- END FIX ---
            
            if x.size == 0:
                x = x_temp
            else:
                x = np.concatenate((x, x_temp), axis = 0)
                # --- FIX: Removed problematic np.unique call ---
                # x = np.unique(x, axis = 0)
                # --- END FIX ---
    else:
        # --- FIX 1: Added low_memory=False to resolve DtypeWarning ---
        dataset = pd.read_csv(path, low_memory=False)

        # --- START FIX: Clean data types *before* filtering ---
        # 1. Fill NaNs *first*
        dataset = dataset.fillna(-1) 
        
        # 2. Force 'is_attack' to numeric, handling errors
        # This converts any remaining strings (like "") to NaN, which we then fill with -1.
        if 'is_attack' in dataset.columns:
            dataset['is_attack'] = pd.to_numeric(dataset['is_attack'], errors='coerce').fillna(-1).astype(int)
        # --- END FIX ---
    
        if mode == 1 or mode == 2:
            dataset = dataset.loc[dataset['is_attack'] == is_attack]
           
        if mode == 0:
            # --- FIX 2: Removed extra [] brackets ---
            dataset.drop(columns=columns_to_drop_packet, inplace = True, errors='ignore')
            dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='mqtt')))]
        elif mode == 1:
            dataset.drop(columns = columns_to_drop_uni, inplace = True, errors='ignore')
        elif mode == 2:
            dataset.drop(columns = columns_to_drop_bi, inplace = True, errors='ignore')
        
        if verbose:                 
            print(dataset.columns)
               
        if mode == 0:
            # --- FIX 3: Added .copy() to prevent SettingWithCopyWarning ---
            x = dataset.loc[dataset['is_attack'] == is_attack].copy()   
            x.drop('is_attack', axis=1, inplace=True)

            # --- START NEW FIX: Force all feature columns (except protocol) to numeric ---
            numeric_cols = x.columns[1:]
            for col in numeric_cols:
                x[col] = pd.to_numeric(x[col], errors='coerce').fillna(-1)
            # --- END NEW FIX ---

            # --- FIX 4: Replaced deprecated OneHotEncoder with ColumnTransformer ---
            numeric_features = list(range(1, x.shape[1])) 

            if preprocessor == None:
                preprocessor = ColumnTransformer(
                    transformers=[
                        # [0] is the 'protocol' column
                        ('protocol_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [0]) 
                    ],
                    remainder='passthrough' 
                )
                x_processed = preprocessor.fit_transform(x)
            else:
                x_processed = preprocessor.transform(x)
            
            if hasattr(x_processed, "toarray"):
                x = x_processed.toarray()
            else:
                x = x_processed
            # --- END FIX 4 ---
        else:
            # Mode 1 and 2 are all numeric
            x_df = dataset.loc[dataset['is_attack'] == is_attack].copy()
            x_df.drop('is_attack', axis=1, inplace=True)
            # Force all to numeric just in case
            for col in x_df.columns:
                x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(-1)
            x = x_df.values
    
    with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
        f.write('all, {}, {} \n'.format(path, x.shape[0]))
    
    # --- FIX: Removed problematic np.unique call ---
    # x = np.unique(x, axis = 0)
    # --- END FIX ---

    with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
        # This count is no longer "unique", it's "total processed"
        f.write('processed, {}, {} \n'.format(path, x.shape[0]))
    
    if (mode == 1 and x.shape[0] > 100000) or (mode == 2 and x.shape[0] > 50000):
            temp = x.shape[0] // 10
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
    elif mode == 0:
        if x.shape[0] > 15000000:
            temp = x.shape[0] // 400
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
        elif x.shape[0] > 10000000:
            temp = x.shape[0] // 200
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
        elif x.shape[0] > 100000:
            temp = x.shape[0] // 10
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))

            
    y = np.full(x.shape[0], label)
    
    with open(os.path.join(folder_name, 'instances_count.csv'),'a') as f:
        f.write('slice, {}, {} \n'.format(path, x.shape[0]))
        
    return x, y

# --- 2. MODIFIED FUNCTION TO ACCEPT 'models_dir' and 'prefix' ---
def classify_sub(classifier, x_train, y_train, x_test, y_test, cm_file_name, summary_file_name, models_dir, classifier_name, prefix, verbose = True):
    classifier.fit(x_train, y_train)
    
    # --- 3. ADDED SAVE LOGIC ---
    if models_dir:
        # Clean classifier name for saving
        model_filename = f"{prefix}_{classifier_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(classifier, model_path)
        if verbose:
            print(f"Model saved to: {model_path}")
    # --- END SAVE LOGIC ---
            
    pred = classifier.predict(x_test)
    
    cm = pd.crosstab(y_test, pred)
    cm.to_csv(cm_file_name)    
    
    # --- FIX: Added zero_division=0 to suppress UndefinedMetricWarning ---
    pd.DataFrame(classification_report(y_test, pred, output_dict = True, zero_division=0)).transpose().to_csv(summary_file_name)
    
    if verbose:
        print(classifier_name + ' Done.\n')
    
    del classifier
    del pred
    del cm
    
def classify(random_state, x_train, y_train, x_test, y_test, folder_name, prefix = "", verbose = True):
    confusion_matrix_folder = os.path.join(folder_name, 'Confusion_Matrix/') 
    summary_folder =  os.path.join(folder_name, 'Summary/') 
    
    # --- 4. DEFINE MODELS DIRECTORY ---
    models_dir = os.path.join(folder_name, "Trained_Models")
    os.makedirs(models_dir, exist_ok=True)
    # --- END DEFINE ---

    os.makedirs(confusion_matrix_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)
            
    # --- 5. PASS 'models_dir' and 'prefix' TO ALL CALLS ---
    
    # 1- Linear
    linear_classifier = LogisticRegression(random_state = random_state, max_iter=1000) # Increased max_iter
    classify_sub(linear_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_linear.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_linear.csv'),
                 models_dir, 'Linear', prefix, verbose)
       
    # 2- KNN
    knn_classifier = KNeighborsClassifier()
    classify_sub(knn_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_knn.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_knn.csv'),
                 models_dir, 'KNN', prefix, verbose)
    
    #3- RBF SVM
    kernel_svm_classifier = SVC(kernel = 'rbf', random_state = random_state, gamma='scale')
    classify_sub(kernel_svm_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_kernel_svm.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_kernel_svm.csv'),
                 models_dir, 'SVM_RBF_Kernel', prefix, verbose)
    
    #4- Naive Bayes
    naive_classifier = GaussianNB()
    classify_sub(naive_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_naive.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_naive.csv'),
                 models_dir, 'Naive', prefix, verbose)

    #5- Decision Tree
    decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = random_state)
    classify_sub(decision_tree_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_decision_tree.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_decision_tree.csv'),
                 models_dir, 'Decision_Tree', prefix, verbose)
    
    #6- Random Forest
    random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = random_state)
    classify_sub(random_forest_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_random_forest.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_random_forest.csv'),
                 models_dir, 'Random_Forest', prefix, verbose)

    # 7- Linear SVM 
    svm_classifier = LinearSVC(random_state = random_state, max_iter=2000, dual=False) # Increased max_iter, set dual=False
    classify_sub(svm_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 os.path.join(confusion_matrix_folder, prefix + '_cm_svm.csv'), 
                 os.path.join(summary_folder, prefix + '_summary_svm.csv'),
                 models_dir, 'SVM_Linear_Kernel', prefix, verbose)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = int, default = 2, help="0: packet, 1: uniflow, 2: biflow")
    parser.add_argument('--output', default='Classification_Bi')
    parser.add_argument('--verbose', type = str2bool, default = True)

    args = parser.parse_args()

    # --- 6. NEW: Define the main 'Outputs' directory ---
    base_output_dir = "Outputs"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # The --output argument now defines the *subfolder* within 'Outputs'
    # e.g., 'Outputs/Bidirectional_Results'
    main_output_dir = os.path.join(base_output_dir, args.output)
    # --- END NEW ---
    
    os.makedirs(main_output_dir, exist_ok=True)
    if args.verbose:
        print(f"Main results directory created at: {os.path.abspath(main_output_dir)}")

    # Resolve dataset folder & filenames based on mode and the repo's folder layout
    if args.mode == 0:
        data_dir = os.path.join('datasets', 'packet_features') # Using lowercase 'datasets'
        files = [
            ('normal.csv', 0, 0),
            ('scan_A.csv', 1, 1),
            ('scan_sU.csv', 1, 2),
            ('sparta.csv', 1, 3),
            ('mqtt_bruteforce.csv', 1, 4)
        ]
    elif args.mode == 1:
        data_dir = os.path.join('datasets', 'uniflow_features') # Using lowercase 'datasets'
        files = [
            ('uniflow_normal.csv', 0, 0),
            ('uniflow_scan_A.csv', 1, 1),
            ('uniflow_scan_sU.csv', 1, 2),
            ('uniflow_sparta.csv', 1, 3),
            ('uniflow_mqtt_bruteforce.csv', 1, 4)
        ]
    else:
        data_dir = os.path.join('datasets', 'biflow_features') # Using lowercase 'datasets'
        files = [
            ('biflow_normal.csv', 0, 0),
            ('biflow_scan_A.csv', 1, 1),
            ('biflow_scan_sU.csv', 1, 2),
            ('biflow_sparta.csv', 1, 3),
            ('biflow_mqtt_bruteforce.csv', 1, 4)
        ]

    # verify dataset folder exists
    if not os.path.isdir(data_dir):
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"[ERROR] Dataset directory not found. Expected it at: {os.path.abspath(data_dir)}. Please check your folder structure.")

    for slice_number in range(10):
        if args.verbose:
            print('Starting Slice #: {}'.format(slice_number))
            print('Start Classification')

        random_state = 0
        
        # 2. Create a sub-folder for this slice *inside* the main output directory
        slice_folder_name = f"Slice_{slice_number}"
        folder_name = os.path.join(main_output_dir, slice_folder_name)
        os.makedirs(folder_name, exist_ok=True)

        # Load first file to initialize x, y
        first = True
        
        # Reset the global preprocessor for each slice
        preprocessor = None
        
        for filename, is_attack, label in files:
            full_path = os.path.join(data_dir, filename)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"[ERROR] Expected dataset file not found: {full_path}")
            
            if args.verbose:
                print(f"Loading file: {full_path}")
                
            x_temp, y_temp = load_file(full_path,
                                       args.mode,
                                       is_attack,
                                       label,
                                       folder_name, 
                                       slice_number,
                                       args.verbose)
            if first:
                x, y = x_temp, y_temp
                first = False
            else:
                x = np.concatenate((x, x_temp), axis=0)
                y = np.append(y, y_temp)
            del x_temp, y_temp

        if args.verbose:
            print(f"Total instances loaded for slice {slice_number}: {x.shape[0]}")
            
        # --- START SCALING (NO-CV RUN) ---
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size = 0.25,
                                                            random_state = 42)
        
        if args.verbose:
            print("Scaling data for no-CV run...")
        
        if args.mode == 0 and preprocessor is not None:
            # Find the split point
            if len(preprocessor.transformers_[0][1].categories_) > 0 and preprocessor.transformers_[0][1].categories_[0].shape[0] > 0:
                n_features_categorical = preprocessor.transformers_[0][1].categories_[0].shape[0]
            else:
                try:
                    n_features_categorical = len(preprocessor.named_transformers_['protocol_encoder'].get_feature_names_out(['protocol']))
                except AttributeError: # Fallback for very old sklearn
                    n_features_categorical = preprocessor.transformers_[0][1].categories_[0].shape[0]

            # Scale numeric features only
            x_train_cat = x_train[:, :n_features_categorical]
            x_train_num = x_train[:, n_features_categorical:]
            x_test_cat = x_test[:, :n_features_categorical]
            x_test_num = x_test[:, n_features_categorical:]
            
            scaler = StandardScaler()
            x_train_num_scaled = scaler.fit_transform(x_train_num)
            x_test_num_scaled = scaler.transform(x_test_num)
            
            # Recombine
            x_train_scaled = np.hstack((x_train_cat, x_train_num_scaled))
            x_test_scaled = np.hstack((x_test_cat, x_test_num_scaled))
        else:
            # Mode 1 and 2 are all numeric, can scale directly
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
        
        # --- 6. SAVE THE SCALER/PREPROCESSOR ---
        scaler_path = os.path.join(folder_name, f"slice_{slice_number}_no_cv_SCALER.joblib")
        joblib.dump(scaler, scaler_path)
        if args.mode == 0 and preprocessor is not None:
            preprocessor_path = os.path.join(folder_name, f"slice_{slice_number}_no_cv_PREPROCESSOR.joblib")
            joblib.dump(preprocessor, preprocessor_path)
        # --- END SAVE ---

        if args.verbose:
            print("Starting classification (no cross-validation)...")
            
        classify(random_state, x_train_scaled, y_train, x_test_scaled, y_test,
                 folder_name, 
                 "slice_{}_no_cross_validation".format(slice_number), args.verbose)
        
        del x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler # Memory cleanup

        if args.verbose:
            print("Starting classification (with 5-fold cross-validation)...")
            
        kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        counter = 0
        for train_idx, test_idx in kfold.split(x, y):
            if args.verbose:
                print(f"--- Fold {counter+1}/5 ---")
            
            x_train_fold, x_test_fold = x[train_idx], x[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # --- START SCALING (CV LOOP) ---
            if args.verbose:
                print("Scaling data for CV fold...")
            
            if args.mode == 0 and preprocessor is not None:
                if len(preprocessor.transformers_[0][1].categories_) > 0 and preprocessor.transformers_[0][1].categories_[0].shape[0] > 0:
                     n_features_categorical = preprocessor.transformers_[0][1].categories_[0].shape[0]
                else:
                     n_features_categorical = len(preprocessor.named_transformers_['protocol_encoder'].get_feature_names_out(['protocol']))

                # Split data
                x_train_cat_fold = x_train_fold[:, :n_features_categorical]
                x_train_num_fold = x_train_fold[:, n_features_categorical:]
                x_test_cat_fold = x_test_fold[:, :n_features_categorical]
                x_test_num_fold = x_test_fold[:, n_features_categorical:]

                # Scale numeric features
                scaler_fold = StandardScaler()
                x_train_num_fold_scaled = scaler_fold.fit_transform(x_train_num_fold)
                x_test_num_fold_scaled = scaler_fold.transform(x_test_num_fold)
                
                # Recombine
                x_train_fold_scaled = np.hstack((x_train_cat_fold, x_train_num_fold_scaled))
                x_test_fold_scaled = np.hstack((x_test_cat_fold, x_test_num_fold_scaled))
            else:
                # Mode 1 and 2 are all numeric
                scaler_fold = StandardScaler()
                x_train_fold_scaled = scaler_fold.fit_transform(x_train_fold)
                x_test_fold_scaled = scaler_fold.transform(x_test_fold)
            
            # --- 7. SAVE THE SCALER/PREPROCESSOR FOR THIS FOLD ---
            scaler_path = os.path.join(folder_name, f"slice_{slice_number}_k_{counter}_SCALER.joblib")
            joblib.dump(scaler_fold, scaler_path)
            if args.mode == 0 and preprocessor is not None:
                preprocessor_path = os.path.join(folder_name, f"slice_{slice_number}_k_{counter}_PREPROCESSOR.joblib")
                joblib.dump(preprocessor, preprocessor_path) # preprocessor is fit once per slice, which is fine
            # --- END SAVE ---
            
            # --- END SCALING ---

            classify(random_state, x_train_fold_scaled, y_train_fold, x_test_fold_scaled, y_test_fold,
                     folder_name, 
                     "slice_{}_k_{}".format(slice_number, counter), args.verbose)
            
            del x_train_fold, x_test_fold, y_train_fold, y_test_fold, x_train_fold_scaled, x_test_fold_scaled, scaler_fold # Memory cleanup
            counter += 1

        # cleanup
        del x
        del y
        
        if args.verbose:
            print(f"--- Slice {slice_number} Complete ---")