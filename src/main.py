from src.dataset import load_housing_data
from src.dataset import split_train_test_ver_1, split_train_test_ver_2, split_train_test_by_id


def main():
    # Download the data need for machine learning problem
    data = load_housing_data()

    print("Data loaded: \n")
    print(
        f"Data has {data.ndim} dimensions, {data.shape} rows and columns,\n"
        f" {data.columns} are columns names.\n\n"
    )
    
    test_ratio = .2
    print("Split data with algorithm version 1: \n")
    df_1, df_2 = split_train_test_ver_1(data, test_ratio)
    print(
        f" {len(df_1)} train + {len(df_2)} test"
        f" The first dataset has {df_1.ndim} dimensions, {df_1.shape} rows and columns,\n"
        f" The sample are:\n {df_1.head()}\n"
        f" The second dataset has {df_2.ndim} dimensions, {df_2.shape} rows and columns,\n"
        f" The sample are:\n {df_2.head()}\n\n"
    )
    
    print("Split data with algorithm version 2: \n")
    df_1, df_2 = split_train_test_ver_2(data, test_ratio, 15)
    print(
        f" {len(df_1)} train + {len(df_2)} test"
        f" The first dataset has {df_1.ndim} dimensions, {df_1.shape} rows and columns,\n"
        f" The sample are:\n {df_1.head()}\n"
        f" The second dataset has {df_2.ndim} dimensions, {df_2.shape} rows and columns,\n"
        f" The sample are:\n {df_2.head()}\n\n"
    )
    
    data_with_id = data.reset_index()
    print("Split data with algorithm version 3: \n")
    df_1, df_2 = split_train_test_by_id(data_with_id, test_ratio, "index")
    print(
        f" The first dataset has {df_1.ndim} dimensions, {df_1.shape} rows and columns,\n"
        f" The sample are:\n {df_1.head()}\n"
        f" The second dataset has {df_2.ndim} dimensions, {df_2.shape} rows and columns,\n"
        f" The sample are:\n {df_2.head()}\n\n"
    )
    
    data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
    print("Split data with algorithm version 3 bis: \n")
    df_1, df_2 = split_train_test_by_id(data_with_id, test_ratio, "id")
    print(
        f" The first dataset has {df_1.ndim} dimensions, {df_1.shape} rows and columns,\n"
        f" The sample are:\n {df_1.head()}\n"
        f" The second dataset has {df_2.ndim} dimensions, {df_2.shape} rows and columns,\n"
        f" The sample are:\n {df_2.head()}\n\n"
    )


if __name__ == "__main__":
    main()
