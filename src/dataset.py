from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import tarfile
import urllib.request
from src.config import TARBALL_PATH, URL_DATA

def load_housing_data():
    tarball_path = Path(TARBALL_PATH) # weset the variable as a path
    if not tarball_path.is_file(): # check if the file doesn't exits, that means we have not yet already download the dataset
        urllib.request.urlretrieve(URL_DATA, tarball_path) # we retrieve it from the location to the path (data/raw)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="data/external") # we extract data to data/external
    else:
        if not Path("data/external/").is_relative_to("housing"): # it is already download, we check if it has been extracted, this is the folder will contains the housing folder
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="data/external") # we extract to data/external
        else:
            print(f"Directory {tarball_path} already created and data extracted.") #already extracted
    return pd.read_csv(Path("data/external/housing/housing.csv"))

def split_train_test_ver_1(data, test_ratio):
    """
    This function try to split data in train and test randomnly.

    Args:
        data (pd.dataframe): the data to be split
        test_ratio (real): the proportion of test data

    Returns:
        Union[pd.dataframe, pd.dataframe]: two datsets, train and test ones
        
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

# the limitation of this version is that if you repeat many times you  will get differents data set
# to avoid it we can use a parameter to specify the state of randomness

def split_train_test_ver_2(data, test_ratio, seed = 1987):
    """
    This function try to split data in train and test randomnly.

    Args:
        data (pd.dataframe): the data to be split
        test_ratio (real): the proportion of test data
        seed (int): the value of the specification randomness

    Returns:
        Union[pd.dataframe, pd.dataframe]: two datsets, train and test ones
        
    """
    np.random.seed(seed=seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

# the two last version will still fail when you get an update data (with new sample (occurence))
# the solution is to use a hash function on the indice (unique identifier) variable

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[-in_test_set], data.loc[in_test_set]

