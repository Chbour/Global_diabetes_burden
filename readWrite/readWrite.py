import sys
import os
import pandas as pd

def readToList(path):
    """
        Reads file from given path and stores results in list
    """
    ll = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            ll.append(line)

    return ll


def savePandasDFtoFile(df, path):

    if path.endswith(".parquet"):
        df.to_parquet(path, engine="pyarrow")
    elif path.endswith(".csv"):
        df.to_csv(path, sep=";")
    else:
        print("ERROR: Unable to save result. Unknown file extension. Supported formats: .parquet, .csv")
        sys.exit()



def readFile(path, columns=None, sep=";"):

    if columns is not None:
        columns = [item.strip() for item in columns.split(',')]

    print("Local mode: Read file..")
    if path.endswith(".csv"):
        return pd.read_csv(path)
#    elif path.endswith(".parquet"):
    elif ".parquet" in path:
        if path.startswith("hdfs"):
            print("INFO: Set HADOOP_HOME variable to: {}".format("/space/hadoop/hadoop_home"))
            os.environ["HADOOP_HOME"] = "/space/hadoop/hadoop_home"

        return pd.read_parquet(path, engine="pyarrow", columns=columns)

    else:
        print("ERROR: Unsupported format file. Allowed only .parquet or .csv files")
        sys.exit()
