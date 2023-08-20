import pandas as pd
import data_processing as dp

def analysis_train_test(
    data_dir_path: str = "data",
    id_field: str = 'id',
    target_field: str = 'target', 
):
    train_df = dp.read_csv(f"{data_dir_path}/train")
    test_df = dp.read_csv(f"{data_dir_path}/test")

    train_column_names = train_df.columns.tolist()
    test_column_names = test_df.columns.tolist()

    print()
    print(train_df)
    print(train_column_names)
    print()
    print(test_df)
    print(test_column_names)
    print()



