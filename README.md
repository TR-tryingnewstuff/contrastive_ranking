# contrastive_ranking
Test of Contrastive Ranking for Stock Picking


import pandas as pd
# Assuming `df` is your DataFrame

# Step 1: Export DataFrame to Excel
output_file = "your_dataframe.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='VisibleSheet')

# Step 2: Add another sheet and hide it using openpyxl
from openpyxl import load_workbook

# Load the workbook and create a new worksheet
wb = load_workbook(output_file)
ws_hidden = wb.create_sheet(title="HiddenSheet")

# Your code to populate the hidden sheet goes here
# For example: ws_hidden.append(['Some', 'Data', 'Here'])

# Hide the sheet
wb[ws_hidden.title].sheet_state = 'hidden'

# Save the workbook with the modifications
wb.save(output_file)
