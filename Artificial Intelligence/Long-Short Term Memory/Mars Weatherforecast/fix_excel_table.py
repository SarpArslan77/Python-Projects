
import pandas as pd
import numpy as np

read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Mars Weatherforecast/clean_mars_weather.xlsx"

try:
    df = pd.read_excel(read_file_path)
    print("\nFile red successfully!")
    if df.empty:
        print("\nThe Excel sheet is empty.")
except FileNotFoundError:
    print(f"\nError: The file {read_file_path} was not found. Exitting the program.")
    exit()

# Add the missing days as NaN to be fixex later.
new_values: dict = {
    "terrestrial_date" : np.nan, "sol" : np.nan, "ls" : np.nan, "month" : "Month 0",
    "min_temp" : np.nan, "max_temp" : np.nan, "pressure" : np.nan,
}
# Find the indexes of missing numbers.
existing_sols: list = df["sol"].tolist()
full_set: set = set(range(min(existing_sols), max(existing_sols)+1))
missing_sols: list = sorted(full_set - set(existing_sols))
# Create the NaN rows for missing days in Excel.
for s in missing_sols:
    df_top = df.iloc[: s-1]
    df_bottom = df.iloc[s-1 :]
    new_row_df = pd.DataFrame([new_values])
    df = pd.concat([df_top, new_row_df, df_bottom], ignore_index=True)

mars_weather_data: dict = {}
# Turn the excel data into NumPy arrays in a dictionary
for column_name in df.columns:
    # Data is being represented by the sol, so we don't need it as an input to the Neural Network (NN).
    if column_name != "terrestrial_date":
        mars_weather_data[column_name] = df[column_name].to_numpy()

for key_name in list(mars_weather_data.keys()):

    data = mars_weather_data[key_name]

    # Treat the values for ls and month cyclical.
    if key_name == "ls":
        solar_longitude_sin = np.sin(2*np.pi * data / 360)
        solar_longitude_cos = np.cos(2*np.pi * data / 360)
        
        mars_weather_data["ls_sin"] = solar_longitude_sin
        mars_weather_data["ls_cos"] = solar_longitude_cos

        # Delete the key "ls" from the dict, since now it is seperated into two keys.
        mars_weather_data.pop("ls", None)

        # Update the excel file with the new mars_weather_data dic
        df["ls_sin"] = mars_weather_data["ls_sin"]
        df["ls_cos"] = mars_weather_data["ls_cos"]

    elif key_name == "month":
        # Split the month number as a str and turn into int.
        month_data_str: np.ndarray[str] = data.astype(str)
        month_numbers_int: np.ndarray = np.char.replace(month_data_str, "Month ", "").astype(int)

        month_sin = np.sin(2*np.pi * month_numbers_int / 12)
        month_cos = np.cos(2*np.pi * month_numbers_int / 12)

        mars_weather_data["month_sin"] = month_sin
        mars_weather_data["month_cos"] = month_cos

        mars_weather_data.pop("month", None)

        df["month_sin"] = mars_weather_data["month_sin"]
        df["month_cos"] = mars_weather_data["month_cos"]

# Fix sol and month count.
starting_sol_count: int = int(df["sol"].max())
for i in range(0, starting_sol_count):
    # Fix the sol count.
    sol_value = df.loc[i, "sol"]
    if np.isnan(sol_value):
        df.loc[i, "sol"] = i+1
    # Fix the month count.
    #? It doesn't fix 100% correctly, since it just copies the last month
    #?  if there is a transition between gaps, it can assign the wrong month
    #?  nevertheless it should be a good enough solution
    last_month: str = df.loc[i, "month"]
    next_month: str = df.loc[min(i+1, starting_sol_count-1), "month"]
    if next_month == "Month 0":
        df.loc[i+1, "month"] = last_month

for j in range(0, starting_sol_count):
    # Determine the neighbors for the row for averaging process.
    neighbors: list[list[float]] = [
        [] for _ in range(len(df.columns)+1)
    ]
    neighbors_index_count: int = 0 

    for column_index in range(2, len(neighbors)-1):
        data = df.iat[j, column_index]

        # Month value should not be changed, so break the process.
        if (df.columns[column_index] == "month") or ( df.columns[column_index] == "terrestrial_date"): 
            continue
        
        # If the value is not NaN, stop the process.
        if not(np.isnan(data)) and \
            (not((df.columns[column_index] == "month_sin") and (data == 0))) and \
            (not((df.columns[column_index] == "ls_sin") and (data == 0))) and \
            (not((df.columns[column_index] == "month_cos") and (data == 1))) and \
            (not((df.columns[column_index] == "ls_cos") and (data == 1))):
            neighbors_index_count += 1
            continue

        # Find the neighboring values.
        for k in range(max(0, j - 25), j):
            neighboring_data: float = df.iat[k, column_index]
            if not(np.isnan(neighboring_data)):
                neighbors[neighbors_index_count].append(neighboring_data)
        for m in range(j+1, min(starting_sol_count, j+26)):
            neighboring_data: float = df.iat[m, column_index]
            if not(np.isnan(neighboring_data)):
                neighbors[neighbors_index_count].append(neighboring_data)

        # Calculate the mean.
        if neighbors[neighbors_index_count]:
            data = np.nanmean(neighbors[neighbors_index_count])
        # If not neighbors found, set the mean to 0.
        else:
            data = 0
        df.iat[j, column_index] = data

        neighbors_index_count += 1

# Delete the not-needed old columns.
df = df.drop(["terrestrial_date", "ls", "month"], axis=1)

# Save the new excel file.
new_file_path = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Mars Weatherforecast/fixed_mars_weather.xlsx"
df.to_excel(new_file_path, index=False) # index=False prevents saving row numbers.

