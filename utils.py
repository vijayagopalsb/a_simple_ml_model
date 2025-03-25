import pandas as pd

def show_dataset_full(dataset):
    df = pd.DataFrame(dataset)
    # Show all columns
    pd.set_option('display.max_columns', None)

    # Show all rows
    pd.set_option('display.max_rows', None)

    return df


    
