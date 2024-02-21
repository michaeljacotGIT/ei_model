import os

import pandas as pd

from datetime import date, timedelta

from matplotlib import font_manager

import EI_model as ei
from openpyxl import load_workbook

fuelPrices = {
    "MGO" : 828.50,
    "VLSFO" : 633.50,
    "HFO" : 503.0
}

VOCEmmision = {
    "HFO" : 3.2,
    "MDO" : 2.4,
    "LNG" : 1.59,
    "VLFSO" : 3.2
}

NOXEmmision = {
    "HFO" : 75.9,
    "MDO" : 56.71,
    "LNG" : 13.44,
    "VLSFO" : 75.9
}

SOXEmmision = {
    "HFO" : 50.83,
    "MDO" : 1.37,
    "LNG" : 0.03,
    "VLSFO" : 1.37
}

# Define the font path
font_path = 'PublicaSansRound-Md.otf'
font_properties = font_manager.FontProperties(fname=font_path)


def main(run_model = False):
    # path to EI Database
    ei_database_path = r"EI Databases\Environmental Impact Database V8.xlsx"

    # Load the wogggrkbook
    wb = load_workbook(filename=ei_database_path, data_only=True)

    # Load the specific worksheet "EID"
    ws = wb['EID']

    # read the data from the excel sheet into a pandas dataframe
    data = pd.DataFrame(ws.values)
    data.columns = data.iloc[0]
    data = data.iloc[1:]

    # Remove rows from "Totals" row
    totals_row = data[data['Vessel Name'] == 'Totals'].index[0]
    data = data.loc[:totals_row - 1]

    emissionData = pd.read_excel(ei_database_path, sheet_name="Emission Factors")

    emission_factors = dict(zip(emissionData["Fuel Type"], emissionData["Emission Factors"]))

    # start date
    start_date = date(2023, 1, 1)

    # end date
    end_date = date(2025, 1, 1)

    # calculate the number of days in 2023
    delta = end_date - start_date

    # generate the list of days
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(delta.days + 1)]
    # jan to june end YTD

    if run_model:
        results, file_name = ei.environmental_impact_calculation(data, emission_factors, dates, fuelPrices, VOCEmmision,SOXEmmision, NOXEmmision, False, report=True, debug=True)
    else:
        #find the most recent file in the "Model Results" folder and read it into a dataframe
        # List all files in the directory
        files = [os.path.join("Model Results", f) for f in os.listdir("Model Results") if
                 os.path.isfile(os.path.join("Model Results", f))]

        if not files:
            print("No files found in the folder.")
        else:
            # Find the most recent file
            most_recent_file = max(files, key=os.path.getctime)
            print(f"Most recent file found: {most_recent_file}")

            # Read the most recent file into a dataframe
            try:
                results = pd.read_csv(most_recent_file)  # Use pd.read_excel() if it's an Excel file
                print("File successfully read into a dataframe.")
            except Exception as e:
                print(f"Error reading the file into a dataframe: {e}")



    # read in the model results
    #df = pd.read_csv(rf"Model Results/{file_name}")

    ei.dashboard(results, background_image_path="Dashboard Backgrounds/background4.png",show = False)

if __name__ == "__main__":
    main(run_model=True)


