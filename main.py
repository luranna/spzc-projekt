import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

base_path="/home/ubuntu/Desktop/spzc/data/"

kddcup_training_set = base_path+'KDDTrain+.txt'
file_path_training_set_scan = base_path+'scan_data.csv'

kddcup_test_set = base_path+'KDDTest+.txt'

tii_src_training_set=base_path+'ttl-infogathering-1.csv'
tii_src_test_set=base_path+'ttl-infogathering-2.csv'
tii_benign_data = base_path+'tii-benign-data.csv'

def prepare_wireshark_dataset_benign(df):
    columns = ['src_ip', 'dst_ip', 'ip_version', 'protocol', 'ip_flags', 'src_port', 'dst_port','tcp_flags', 'src_bytes', 'dst_bytes']
    new_df=df.loc[:,columns]
    df_final=normal_traffic_classification(new_df)
    return df_final

def prepare_wireshark_dataset_scan(df):
    columns = ['src_ip', 'dst_ip', 'ip_version', 'protocol', 'ip_flags', 'src_port', 'dst_port','tcp_flags', 'src_bytes', 'dst_bytes']
    new_df=df.loc[:,columns]
    df_final=scan_data_classification(new_df)
    return df_final


def prepare_kddcup_dataset(df):
    columns = (['duration','protocol','service','tcp_flags','src_bytes','dst_bytes','land','wrong_fragment'
    ,'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells'
    ,'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
    ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
    ,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
    ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
    ,'attack', 'level'])
    normal_flow_and_probe_attacks = ['normal','ipsweep','mscan','nmap','portsweep','saint','satan']
    drop_columns=['land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells'
    ,'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
    ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
    ,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
    ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level']

    df.columns = columns

    df_final = df.loc[df["attack"].isin(normal_flow_and_probe_attacks)]
    df_final= add_anomaly_classification(df_final)
    df_final=df_final.drop(columns=drop_columns)
    return df_final

def add_anomaly_classification(df):
    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    df['class_num'] = is_attack
    return df

#add classification of scan network flow from Wireshark (1)
def scan_data_classification(df):
    df['class_num'] = 1
    return df

#add classification of normal network flow from Wireshark (0)
def normal_traffic_classification(df):
    df['class_num'] = 0
    return df

def preprocess_data(df_train, df_test):
    features_to_encode = ['src_ip', 'dst_ip','protocol', 'ip_flags','tcp_flags','class_num']
    numeric_features = ['src_bytes', 'dst_bytes','src_port', 'dst_port']
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
    
    train_set=train_set.fillna(0)
    test_set=test_set.fillna(0)
    return train_set, test_set

def train_and_test_model(train_set, test_set):
    train_y_data = train_set.class_num.values # wartości klasyfikacji - zbiór treningowy
    train_set = train_set.drop(["class_num"], axis=1)  # dataframe z wyodrębnionymi cechami do badania - zbiór treningowy

    test_y_data = test_set.class_num.values  # wartości klasyfikacji - zbiór testowy
    test_set = test_set.drop(["class_num"], axis=1) # dataframe z wyodrębnionymi cechami do badani - zbiór testowy

    model = LogisticRegression(max_iter=500)
    model.fit(train_set, train_y_data)
    multi_predictions = model.predict(test_set)

    print("Model accuracy score:")
    print(accuracy_score(multi_predictions,test_y_data))

# function for joining kdd and wireshark as they have different columns
def join_datasets(df,df2):
    df_final=pd.DataFrame()
    df_final=df_final._append(df, ignore_index=True)
    df_final=df_final._append(df2, ignore_index=True)
    return df_final

def prepare_data():
    df = pd.read_csv(kddcup_training_set)
    df2 = pd.read_csv(tii_src_training_set)
    df3= pd.read_csv(file_path_training_set_scan)
    df4=pd.read_csv(tii_benign_data)
    test_df = pd.read_csv(kddcup_test_set)
    test_df2=pd.read_csv(tii_src_test_set)

    #training data
    kdd_df= prepare_kddcup_dataset(df)
    wireshark_df=prepare_wireshark_dataset_scan(df2)
    wireshark_df2=prepare_wireshark_dataset_scan(df3)
    wireshark_df=wireshark_df._append(wireshark_df2, ignore_index=True)
    final_train_set=join_datasets(kdd_df,wireshark_df)
    benign_traffic=prepare_wireshark_dataset_benign(df4)
    final_train_set=join_datasets(final_train_set,benign_traffic)

    #test data
    kdd_test_set=prepare_kddcup_dataset(test_df)
    wireshark_test_set=prepare_wireshark_dataset_scan(test_df2)
    final_test_set=join_datasets(kdd_test_set,wireshark_test_set)

    return final_train_set, final_test_set

def main():
    final_train_set, final_test_set=prepare_data()
    preprocessed_final_train_set, preprocessed_final_test_set=preprocess_data(final_train_set, final_test_set)
    train_and_test_model(preprocessed_final_train_set, preprocessed_final_test_set)
    

if __name__ == "__main__":
    main()
