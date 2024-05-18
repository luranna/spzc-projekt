import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

base_path="/home/ubuntu/Desktop/spzc/data/"

file_path_training_set = base_path+'KDDTrain+.txt'
file_path_training_set2 = base_path+'Train_data.csv'
file_path_training_set_scan = base_path+'scan_data.csv'

file_path_test = base_path+'KDDTest+.txt'
file_path_test2 = base_path+'Test_data.csv'


def prepare_datasets(df, df2):
    columns = (['duration'
    ,'protocol_type'
    ,'service'
    ,'flag'
    ,'src_bytes'
    ,'dst_bytes'
    ,'land'
    ,'wrong_fragment'
    ,'urgent'
    ,'hot'
    ,'num_failed_logins'
    ,'logged_in'
    ,'num_compromised'
    ,'root_shell'
    ,'su_attempted'
    ,'num_root'
    ,'num_file_creations'
    ,'num_shells'
    ,'num_access_files'
    ,'num_outbound_cmds'
    ,'is_host_login'
    ,'is_guest_login'
    ,'count'
    ,'srv_count'
    ,'serror_rate'
    ,'srv_serror_rate'
    ,'rerror_rate'
    ,'srv_rerror_rate'
    ,'same_srv_rate'
    ,'diff_srv_rate'
    ,'srv_diff_host_rate'
    ,'dst_host_count'
    ,'dst_host_srv_count'
    ,'dst_host_same_srv_rate'
    ,'dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate'
    ,'dst_host_srv_diff_host_rate'
    ,'dst_host_serror_rate'
    ,'dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate'
    ,'dst_host_srv_rerror_rate'
    ,'attack'
    ,'level'])

    df.columns = columns
    df = df.iloc[:, :-1]

    is_attack = df.attack.map(lambda a: "normal" if a == 'normal' else "anomaly")

    df = df.iloc[:, :-2]

    df['class'] = is_attack

    df_final = df._append(df2, ignore_index=True)
    df_final.head()
    return df

def add_anomaly_classification(df):
    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    df['class_num'] = is_attack
    return df

#dane ze skanowania zarejestrowane przez program wireshark mają inną strukturę niż te z datasetów, plus wszystkie zarejestrowane pakiety są powiązane ze skanami - stąd wszędzie trzeba dać klasyfikację 1
def scan_data_prepraration(scan_df):
    scan_df['class_num'] = 1
    return scan_df.head()

def join_dataset_and_scan_data(df, scan_df):
    #joining the same columns from both df that will be later used for model training
    df_new =df['protocol_type', 'flag', 'src_bytes', 'dst_bytes','class_num']
    df_scan_new=scan_df['protocol_type', 'flag', 'src_bytes', 'dst_bytes','class_num']
    df_final = df_new._append(df_scan_new, ignore_index=True)
    return df_final


def preprocess_data(df_train, df_test):
    features_to_encode = ['protocol_type', 'flag', 'class']
    numeric_features = ['src_bytes', 'dst_bytes']
    # One-hot encoding of categorical features

    features_encoded = pd.get_dummies(df_train[features_to_encode], drop_first=True)
    test_features_encoded = pd.get_dummies(df_test[features_to_encode], drop_first=True)
    
    # Ensure both train and test sets have the same columns
    test_index = np.arange(len(df_test.index))
    column_diffs = list(set(features_encoded.columns.values) - set(test_features_encoded.columns.values))
    diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)
    
    column_order = features_encoded.columns.to_list()
    test_encoded_temp = test_features_encoded.join(diff_df)
    test_final = test_encoded_temp[column_order].fillna(0)
    
    # Join the numeric features
    train_set = features_encoded.join(df_train[numeric_features])
    test_set = test_final.join(df_test[numeric_features])
    
    return train_set, test_set

def train_and_test_model(train_set, test_set):
    train_y_data = train_set.class_normal.values # wartości klasyfikacji - zbiór treningowy
    train_set = train_set.drop(["class_normal"], axis=1)  # dataframe z wyodrębnionymi cechami do badania - zbiór treningowy

    test_y_data = test_set.class_normal.values  # wartości klasyfikacji - zbiór testowy
    test_set = test_set.drop(["class_normal"], axis=1) # dataframe z wyodrębnionymi cechami do badani - zbiór testowy

    model = LogisticRegression(max_iter=500)
    model.fit(train_set, train_y_data)
    multi_predictions = model.predict(test_set)

    print("Model accuracy score:")
    print(accuracy_score(multi_predictions,test_y_data))

def main():
    #data preparation
    df = pd.read_csv(file_path_training_set)
    df2 = pd.read_csv(file_path_training_set2)
    test_df = pd.read_csv(file_path_test)
    test_df2 = pd.read_csv(file_path_test2)
    
    train_datasets_combined = prepare_datasets(df, df2)
    test_datasets_combined = prepare_datasets(test_df, test_df2)

    scan_df = pd.read_csv(file_path_training_set_scan)
    #final_train_data = join_dataset_and_scan_data(train_datasets_combined, scan_df)

    final_train_set, final_test_set = preprocess_data(train_datasets_combined, test_datasets_combined)
    
    #training and testing of the model
    train_and_test_model(final_train_set, final_test_set)
    

if __name__ == "__main__":
    main()
