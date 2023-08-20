import sqlite3
from sqlite3 import Error
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import sys
import argparse

matplotlib.use('Agg')

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def load_data(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return: pandas Dataframe
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM filialdata")
    df = pd.read_sql("SELECT * FROM filialdata", conn)
    return df

def preprocess(df):
    """
    Preprocess data by dropping the given columns and also checking for missing values
    :param df: given dataframe
    :return preprocessed DataFrame
    """

    df_dropped = df.drop(["KaufKraft_Lebensmittelhandel_KOPF", "KaufKraft_Drogeriefachhandel_KOPF", "Kaufkraft_KOPF", "ANZAHL_Verbrauchermärkte_im Umkreis von 20min", "ANZAHL_Diskonter_im Umkreis von 20min"], axis=1)
    df_dropped.set_index("FILIALE", inplace = True)
    df_withna = df_dropped[df_dropped.isnull().any(axis=1)]
    if not df_withna.empty:
        # Here also we can fill NANs from grouped by B_LAND data with mean values.
        print(df_withna.head())
        df_dropped.fillna(df.mean(), axis=1, inplace=True)
    return df_dropped

def cluster_data(df, train_size=0.8, eps=0.5, min_samples=5):
    """
    Splitting data and clustering with given methods UMAP and DBSCAN
    :param df: given dataframe
    :return preprocessed DataFrame
    """    
    print(f"Passed params epsilon: {eps} and min_samples: {min_samples}")
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop("B_LAND", axis=1), df["B_LAND"], train_size=train_size)
    
    ump = umap.UMAP(random_state=42)
    umap_data = ump.fit_transform(X_train)
    
    plt.figure(figsize=(20,12))
    plt.title("UMAP reduction plot")
    sns.scatterplot(x=umap_data[:,0], y=umap_data[:,1], hue=Y_train, palette='cool')
    plt.savefig('img/UMAP.png')
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(umap_data)
    plt.figure(figsize=(20,12))
    plt.title(f"DBSCAN using the umap data with params epsilon: {eps} and min_samples: {min_samples}")
    sns.scatterplot(x=umap_data[:,0], y=umap_data[:,1], hue=dbscan.labels_, palette='Set2')
    plt.savefig('img/DBSCAN.png')

    print("Image DBSCAN and UMAP are being created in folder img")

def main(eps=0.5, min_samples=5):

    df = None
    database = "filialdata.db"
    df_cols = ['B_LAND', 'VERKAUFS_M2', 'Umsatz', 'Kund. Anz.', 'Aktionsanteil',
       'Anteil Clever', 'Anteil Ja!Natürlich', 'Anteil Feinkost',
       'Anteil Obst&Gemüse', 'Anteil BILLA Marke', 'Anteil BILLA Corso',
       'Anteil Getränke ohne Alkohol', 'Anteil Alkohol',
       'ANZAHL_Verbrauchermärkte_im Umkreis von 20min',
       'ANZAHL_Diskonter_im Umkreis von 20min',
       'KaufKraft_Drogeriefachhandel_KOPF',
       'KaufKraft_Lebensmittelhandel_KOPF', 'Kaufkraft_KOPF']
    
    # create a database connection
    conn = create_connection(database)

    with conn:
        df = load_data(conn)
        print(f"The shape of the dataset is: {df.shape}")

    all_exist = all(col in df.columns for col in df_cols)

    if all_exist:
        print("All colums exist")
    else:
        raise ValueError("Some colums are missing from the original dataset")
    
    cleaned_data = preprocess(df)
    cluster_data(cleaned_data, eps=eps, min_samples=min_samples)

if __name__ == '__main__':
    
    print("Usage: python my_script.py <arg1: epsilon value> <arg2: min_samples for DBSCAN>")
    
    # Get the command-line arguments
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
    else:
        arg1 = 0.5
    
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
    else:
        arg2 = 5
    main(arg1, arg2)