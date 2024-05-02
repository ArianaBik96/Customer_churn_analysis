import os
import pandas as pd

# Get the current directory of the script
current_directory = os.path.dirname(__file__)
# Define the relative path to the CSV file
csv_file_path = os.path.join(current_directory, "..", "data", "BankChurners.csv")
# Read the CSV file using the relative path
churners_df = pd.read_csv(csv_file_path, index_col='CLIENTNUM', sep=',')

