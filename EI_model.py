import pandas as pd
from datetime import datetime
from openpyxl import load_workbook
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import bar_chart_race as bcr
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
from PIL import Image
from bs4 import BeautifulSoup
import requests
import re





"""
MODEL AXIOMS 

 1 . Assumes the vessel fuel rate entered was the average fuel rate before and after application of the coating.
 
 2 . Assumes that immediately after dry dock, the vessel will continue operating at that design speed with a 0.84% degradation 
      in fuel efficiency per year. After 5 years, the vessel will be dry docked, our coating will remain intact or be repaired to 
      standards, and a refloat will occur. The vessel will recur the 0.84% per year over the 5 year period and regain 4.6% fuel efficiency.
 
 3. Assumes the activity column impacts both the pre and post consumption equally.
 
 4. Assumes that only XGIT-FUEL prevents the avoidance of Cu, VOC, and paint. Propeller coatings are not currently widely adopted, so we 
      cant assume that we would be saving against a biocidal coating in terms of these metrics.
      
 5. Assumes a fixed leeching rate of copper at _____g/day/m^2
 
 6. Assumes activity rate is fixed over the simulation.
 
 7. Assumes that the consumption rates provided are averages that do not consider the activity rate of the vessel.
 
 9. Assumes VOC is prevented via the lowered % solids of XGIT-FUEL. Does not consider VOC saved via emission reduction.
 
 10. Assumes fixed emmision values from 2018 IMO values.
 
 11. EU-ETS Credit savings is pulled from https://tradingeconomics.com/commodity/carbon at the time of the simulation. The savings col in the final
    dataframe assumes 100% of vessels in the fleet operate only within the EU ports. Multiply this value by the average % of vessels that operate
    within the EU to get a more accurate value.

"""

# Define the font path
font_path = 'PublicaSansRound-Md.otf'
font_properties = font_manager.FontProperties(fname=font_path)

# Define a helper function to set font properties for a given axis
def set_axis_font(ax, font_properties):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontproperties(font_properties)

    # If your plot contains additional text elements, you'd update them like this:
    for text in ax.texts:  # This will include any text added using ax.text(...)
        text.set_fontproperties(font_properties)



path = r"C:\Users\jmurp\Graphite Innovation & Technologies\Corporate - Documents\Post-Sales Operations\12-Energy & Efficiency\1 - Performance & Analysis\22 - Environmental Impact\Environmental Impact Database V7.xlsx"


# Load the workbook
wb = load_workbook(filename=path, data_only=True)

# Load the specific worksheet "EID"
ws = wb['EID']

# read the data from the excel sheet into a pandas dataframe
data = pd.DataFrame(ws.values)
data.columns = data.iloc[0]
data = data.iloc[1:]


# Remove rows from "Totals" row
totals_row = data[data['Vessel Name'] == 'Totals'].index[0]
data = data.loc[:totals_row - 1]

emissionData = pd.read_excel(path,sheet_name="Emission Factors")

emission_factors = dict(zip(emissionData["Fuel Type"], emissionData["Emission Factors"]))

#print(emission_factors)

def check_fuel_type(fuel_type, emission_factors):
    # Split the fuel type by '/' and strip any whitespace
    fuels = [f.strip() for f in fuel_type.split('/')]
    for f in fuels:
        if f in emission_factors:  # check if fuel type is a key in the dictionary
            return f
    # If neither fuel type is found, return None
    return None


def get_website_body_as_string(url, debug=False):
    # Read the URL from a json file for security reasons as per user guidelines
    # Make sure to create a json file with the API keys or passwords if needed
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            body_content = soup.body.get_text(separator=' ', strip=True)
            if debug:
                print(f"URL fetched successfully: {url}")
                print(f"Status Code: {response.status_code}")
                print("Body content extracted")
            return body_content
        else:
            if debug:
                print(f"Failed to fetch URL: {url}")
                print(f"Status Code: {response.status_code}")
            return ""
    except Exception as e:
        if debug:
            print(f"An error occurred: {e}")
        return ""

def find_eu_ets_price(text, debug=False):
    # Regex pattern to find "EU Carbon Permits" followed by a price like pattern
    pattern = r"EU Carbon Permits\s+(\d+\.\d+)"

    try:
        # Search for the pattern in the provided text
        match = re.search(pattern, text)

        # If a match is found, return the price
        if match:
            price = match.group(1)
            if debug:
                print(f"EU-ETS Carbon Credit price found: {price}")
            return price
        else:
            if debug:
                print("EU-ETS Carbon Credit price not found.")
            return None
    except Exception as e:
        if debug:
            print(f"An error occurred: {e}")
        return None

def getETUPrice():
    try:
        url = "https://tradingeconomics.com/commodity/carbon"  # Replace with your actual URL
        body_string = get_website_body_as_string(url, debug=True)
        price = find_eu_ets_price(body_string, debug=True)

        if price is not None:
            # Convert price to USD from EUR
            price_usd = round(float(price) * 1.18, 2)
        else:
            # If price is not found, use the default value
            price_usd = 93.37

    except Exception as e:
        print(f"An error occurred: {e}")
        # Use the default value in case of any error
        price_usd = 93.37

    return price_usd

def plot_progress(df):
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Convert total CO2 to million tons
    df['Total CO2 Savings Today (tons)'] = df['Total CO2 Saved Today (t)'] / 1e6

    # Calculations
    current_date = pd.to_datetime(date.today())
    data_to_current = df.loc[df.index <= current_date]
    data_future = df.loc[df.index > current_date]
    goal_CO2 = 4.5  # million tons
    goal_Cu = 1e6  # kg
    current_CO2 = data_to_current['Total CO2 Savings Today (tons)'].iloc[-1]
    current_Cu = data_to_current['Total Cu Savings Today (kg)'].iloc[-1]

    # Create plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    # Plot actual data up to the current date
    ax[0].plot(data_to_current.index, data_to_current['Total CO2 Savings Today (tons)'], label='Actual data')
    ax[1].plot(data_to_current.index, data_to_current['Total Cu Savings Today (kg)'], label='Actual data')

    # Plot predictions
    ax[0].plot(data_future.index, data_future['Total CO2 Savings Today (tons)'], linestyle='--', color='red', label='Prediction')
    ax[1].plot(data_future.index, data_future['Total Cu Savings Today (kg)'], linestyle='--', color='red', label='Prediction')

    # Plot goals
    ax[0].axhline(goal_CO2, color='green', linestyle=':', label='Goal')
    ax[1].axhline(goal_Cu, color='green', linestyle=':', label='Goal')

    # Set titles, labels, and grid
    ax[0].set_title('Progress towards a goal of 4.5 million tons of CO2 Emission by 2030')
    ax[0].set_ylabel('Total CO2 (million tons)')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].set_title('Progress towards a goal of 1 million kg of Cu saved by 2030')
    ax[1].set_ylabel('Total Cu (kg)')
    ax[1].grid(True)
    ax[1].legend()

    # Set the formatter for the y-axis
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax[0].yaxis.set_major_formatter(formatter)
    ax[1].yaxis.set_major_formatter(formatter)

    # Improve layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_total_CO2_savings(data, nLargest=10, debug=False):
    """
    Plot the total daily CO2 savings across the fleet and add annotations for the start dates of CO2 savings for each vessel.

    Parameters:
    - data: A pandas DataFrame that includes the 'Total CO2 Saved Today (t)' column and columns for each vessel's daily CO2 savings.
    - nLargest: Number of top vessels to annotate, based on daily CO2 savings.
    - debug: Boolean to turn on debugging print statements.
    """

    # Convert Date to datetime and set as index
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index('Date', inplace=True)

    # Identify the columns for each vessel's daily CO2 savings
    vessel_columns = [col for col in data.columns if 'CO2 Saved Today (t)' in col and col != 'Total CO2 Saved Today (t)']

    # Find start dates for CO2 savings for each vessel and calculate their total CO2 saved
    vessel_savings = {}
    for vessel_column in vessel_columns:
        vessel_data = data[data[vessel_column] > 0]
        vessel_start_date = vessel_data.index.min()
        total_savings = vessel_data[vessel_column].sum()
        if pd.notnull(vessel_start_date):
            vessel_name = vessel_column.replace(' CO2 Saved Today (t)', '')
            vessel_savings[vessel_name] = (vessel_start_date, total_savings)

    if debug:
        print("Vessel savings:", vessel_savings)

    # Sort vessels by total CO2 saved and take the top nLargest
    top_vessels = sorted(vessel_savings.items(), key=lambda x: x[1][1], reverse=True)[:nLargest]

    if debug:
        print("Top vessels:", top_vessels)

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Total CO2 Saved Today (t)'], label='Total CO2 Saved Today (t)')
    plt.xlabel('Date', fontsize=30)
    plt.ylabel('Daily CO2 Savings (t)', fontsize=30)
    plt.title('Total Daily CO2 Savings Across the Fleet', fontsize=50)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    y_values = np.linspace(data['Total CO2 Saved Today (t)'].min(), data['Total CO2 Saved Today (t)'].max(), len(top_vessels))

    # Annotate top vessels
    texts = []
    for i, (vessel_name, (start_date, _)) in enumerate(top_vessels):
        xy = (start_date, data.loc[start_date, 'Total CO2 Saved Today (t)'])
        text = plt.text(start_date, y_values[i], vessel_name, size=15)
        plt.arrow(start_date, y_values[i], 0, xy[1]-y_values[i], length_includes_head=True, head_width=0.15, head_length=0.15, fc='k', ec='k')
        texts.append(text)

    plt.grid(True)
    plt.show()


def plot_total_money_savings(data):
    """
    Plot the total daily money savings across the fleet and add annotations for the start dates of CO2 savings for each vessel.

    Parameters:
    - data: A pandas DataFrame that includes the 'Total Daily Money Saved ($USD)' column and columns for each vessel's daily CO2 savings.
    """

    # Identify the columns for each vessel's daily CO2 savings
    vessel_columns = [col for col in data.columns if 'CO2 Saved Today (t)' in col and col != 'Total CO2 Saved Today (t)']

    # Find start dates for CO2 savings for each vessel and add annotations
    start_dates = []
    for vessel_column in vessel_columns:
        vessel_start_date = data[data[vessel_column] > 0].index.min()
        if pd.notnull(vessel_start_date):
            start_dates.append((vessel_start_date, vessel_column.replace(' CO2 Saved Today (t)', '')))

    # Sort the start dates
    start_dates.sort()

    # Plot total daily money savings
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Total Daily Money Saved ($USD)'], label='Total Daily Money Saved ($USD)')
    plt.xlabel('Date')
    plt.ylabel('Daily Money Saved ($USD)')
    plt.title('Total Daily Money Savings Across the Fleet')

    # Define evenly spaced y values for the labels
    y_values = np.linspace(data['Total Daily Money Saved ($USD)'].min(), data['Total Daily Money Saved ($USD)'].max(), len(start_dates))

    # Add annotations and store the text objects for later adjustment
    texts = []
    for i, (start_date, vessel_name) in enumerate(start_dates):
        xy = (start_date, data.loc[start_date, 'Total Daily Money Saved ($USD)'])
        text = plt.text(start_date, y_values[i], vessel_name, size=8)
        plt.arrow(start_date, y_values[i], 0, xy[1]-y_values[i], length_includes_head=True, head_width=0.15, head_length=0.15, fc='k', ec='k')
        texts.append(text)

    plt.legend()
    plt.grid(True)
    plt.show()


from typing import Tuple
import datetime as dt


def weeklyReport(df: pd.DataFrame, quarter: int, month: int, week: int, debug=False) -> Tuple[str, str, str, str]:
    # Make sure 'Date' column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the current year
    current_year = dt.datetime.now().year

    # Define the start and end of the quarter
    quarter_start = pd.Timestamp(year=current_year, month=3 * quarter - 2, day=1)
    quarter_end = pd.Timestamp(year=current_year, month=3 * quarter, day=1) + pd.DateOffset(months=1, days=-1)

    # Define the start and end of the month
    month_start = pd.Timestamp(year=current_year, month=month, day=1)
    month_end = pd.Timestamp(year=current_year, month=month, day=1) + pd.DateOffset(months=1, days=-1)

    # Define the start and end of the week within the month
    week_start = month_start + pd.DateOffset(weeks=week-1)
    week_end = week_start + pd.DateOffset(weeks=1, days=-1)

    # Make sure the week_end doesn't exceed the month_end
    week_end = min(week_end, month_end)

    # Define the start and end of the year (YTD)
    ytd_start = pd.Timestamp(year=current_year, day=1, month=1)
    ytd_end = month_end  # End of YTD is same as end of month

    # Create a dictionary to store the start and end dates for each period
    periods = {
        'Q': (quarter_start, quarter_end),
        'M': (month_start, month_end),
        'W': (week_start, week_end),
        'YTD': (ytd_start, ytd_end)
    }

    # Debugging information
    if debug:
        print("Periods:", periods)

    # Initialize an empty dictionary to store the results
    results = {}

    # Calculate the metrics for each period
    for period, (start, end) in periods.items():
        period_df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        co2 = round(period_df['Total CO2 Saved Today (t)'].sum())
        voc = round(period_df['TOTAL VOC SAVED (kg)'].iloc[-1] - period_df['TOTAL VOC SAVED (kg)'].iloc[0])
        nox_sox = round(period_df['Cumulative Avoided NOX and SOX (t)'].iloc[-1] -
                        period_df['Cumulative Avoided NOX and SOX (t)'].iloc[0])
        cu = round(period_df['Total Cu (kg)'].iloc[-1] - period_df['Total Cu (kg)'].iloc[0])
        money_saved = '{:,.2f}'.format(period_df['Total Daily Money Saved ($USD)'].sum())
        results[period] = (co2, voc, nox_sox, cu, money_saved)

    # Define the template string
    template = """
    In {time_period} coated vessels are estimated to have avoided ~{co2} tCO2 emissions, eliminated ~{voc} kg of VOCs, avoided ~{nox_sox} tons of sulfur & nitrogen oxides, and avoided ~{cu} kg of copper from biocides. GIT is working to both reduce atmospheric emissions and protect sensitive marine ecosystem, rich in biodiversity. \n We have saved our clients approximately ${money_saved} USD in this time period
    """

    # Format the template string for each period and return the results
    q_string = template.format(time_period=f'Q{quarter}', co2=results['Q'][0], voc=results['Q'][1],
                               nox_sox=results['Q'][2], cu=results['Q'][3], money_saved=results['Q'][4])
    m_string = template.format(time_period=f'{month_start.strftime("%B")}', co2=results['M'][0], voc=results['M'][1],
                               nox_sox=results['M'][2], cu=results['M'][3], money_saved=results['M'][4])
    w_string = template.format(time_period=f'Week {week} of {month_start.strftime("%B")}', co2=results['W'][0], voc=results['W'][1],
                               nox_sox=results['W'][2], cu=results['W'][3], money_saved=results['W'][4])
    ytd_string = template.format(time_period='1H (YTD)', co2=results['YTD'][0], voc=results['YTD'][1],
                                 nox_sox=results['YTD'][2], cu=results['YTD'][3], money_saved=results['YTD'][4])

    # Print the money saved for each period
    print(f"In Q{quarter} : ${results['Q'][4]} USD")
    print(f"In {month_start.strftime('%B')} : ${results['M'][4]} USD")
    print(f"In Week {week} of {month_start.strftime('%B')} : ${results['W'][4]} USD")
    print(f"YTD : ${results['YTD'][4]} USD")

    return q_string, m_string, w_string, ytd_string




def calculate_monthly_impact(df, year):
    # Convert 'Date' column to datetime if it's not already
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])

    # Filter data for the given year
    df_year = df[df['Date'].dt.year == year]

    # Set 'Date' column as index to resample data by month
    df_year = df_year.set_index('Date')

    # For each month, get the first non-zero values and the last values
    df_year_grouped = df_year.groupby(pd.Grouper(freq='M'))

    monthly_start = df_year_grouped[
        ['Cumulative CO2 Prevented (tons)', 'TOTAL VOC SAVED (kg)', 'Cumulative Avoided NOX and SOX (t)',
         'Total Cu (kg)']].apply(lambda x: x[x > 0].min())
    monthly_end = df_year_grouped[
        ['Cumulative CO2 Prevented (tons)', 'TOTAL VOC SAVED (kg)', 'Cumulative Avoided NOX and SOX (t)',
         'Total Cu (kg)']].last()

    # Subtract start of month values from end of month values for each column
    monthly_impact = monthly_end - monthly_start

    # Print the result
    for i in range(len(monthly_impact)):
        print("For the month of", monthly_impact.index[i].strftime('%B %Y'))
        print("Cumulative CO2 Prevented (tons):", monthly_impact.iloc[i]['Cumulative CO2 Prevented (tons)'])
        print("TOTAL VOC SAVED (kg):", monthly_impact.iloc[i]['TOTAL VOC SAVED (kg)'])
        print("Cumulative Avoided NOX and SOX (t):", monthly_impact.iloc[i]['Cumulative Avoided NOX and SOX (t)'])
        print("Total Cu (kg):", monthly_impact.iloc[i]['Total Cu (kg)'])
        print()

# Redefine the plotting function to save the plot as a high-resolution image
def projectedYearlySavings(df, years, multipliers):
    # Get the total values for 2023
    total_values_2023 = df['TOTAL'].values

    # Create a new dataframe for the projected values
    projected_values = pd.DataFrame(index=df['Indicators'], columns=years)
    projected_values[2023] = total_values_2023

    # Calculate the projected values for the provided years
    for year, multiplier in zip(years[1:], multipliers):
        projected_values[year] = projected_values[year - 1] * multiplier

    # Define the color mapping
    color_mapping = {
        'Avoided Carbon Dioxide Equivalent (tons)': 'black',
        'Avoided Volatile Organic Compounds (kg)': '#1668BD',
        'Avoided Sulfur & Nitrogen Oxides (tons)': '#BBCF2E',
        'Avoided Copper from Biocides (kg)': '#FF8B00',
    }

    # Split the dataframe into two dataframes based on the desired categories
    projected_values1 = projected_values.loc[['Avoided Carbon Dioxide Equivalent (tons)', 'Avoided Volatile Organic Compounds (kg)'], :]
    projected_values2 = projected_values.loc[['Avoided Sulfur & Nitrogen Oxides (tons)', 'Avoided Copper from Biocides (kg)'], :]

    # Plot the projected values in two separate bar graphs with 4 bars side by side for each year
    fig, axs = plt.subplots(1, 2, figsize=(20,7))

    projected_values1.transpose().plot(kind='bar', ax=axs[0], color=[color_mapping[indicator] for indicator in projected_values1.index])
    axs[0].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', prop={'size': 16})
    axs[0].set_title('Projected CO2 and VOC Savings (2023-2026)', fontsize=14)
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Total Value')
    axs[0].set_xticklabels(years, rotation=0)

    projected_values2.transpose().plot(kind='bar', ax=axs[1], color=[color_mapping[indicator] for indicator in projected_values2.index])
    axs[1].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', prop={'size': 16})
    axs[1].set_title('Projected Copper, NOx, and SOx Savings (2023-2026)', fontsize=14)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Total Value')
    axs[1].set_xticklabels(years, rotation=0)

    # Format y-axis labels to include commas
    axs[0].get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )

    axs[1].get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )


    plt.tight_layout()

    # Save the plot as a high-resolution PNG image
    plt.savefig('projected_values.png', dpi=300)
    plt.show()

def adjusted_fuel_consumption(days_since_application, fuel_consumption_pre_application, fuel_consumption_post_application):
    # Calculate the yearly loss in efficiency
    yearly_loss = 0.0084  # 0.84% loss per year

    # Calculate how many full 5-year periods have passed
    five_year_periods = days_since_application // 1825

    # Calculate the number of remaining days after the last complete 5-year period
    remaining_days = days_since_application % 1825

    # Initialize the fuel consumption with the post-application value
    fuel_consumption = fuel_consumption_post_application

    # If a complete 5-year period has passed, the fuel consumption is reset to the post-application value
    if five_year_periods > 0:
        fuel_consumption = fuel_consumption_post_application

    # Add the increase for the remaining days
    fuel_consumption += fuel_consumption * yearly_loss * (remaining_days / 365)
    fuel_consumption = min(fuel_consumption, fuel_consumption_pre_application)

    return fuel_consumption

def environmental_impact_calculation(data, emission_factors, days, fuelPrices,VOCEmmision,SOXEmmision,NOXEmmision, output,report = True,debug = False):
    #Pull EU-ETS current price

    ETSprice = getETUPrice()


    # Define the relevant columns
    relevant_columns = ["Vessel Name", "Date Coated", "Coating Type", "Coated Hull Area (m^2)","Activity %",
                        "Fuel Cons t/day (Pre Application)", "Fuel Cons t/day (Post Application)","Fuel Type"]

    required_columns = ["Vessel Name", "IMO", "Ship Type", "Owner", "Coating Type", "Date Coated",
                        "Coated Hull Area (m^2)", "Activity %", "Fuel Cons t/day (Pre Application)",
                        "Fuel Cons t/day (Post Application)", "Fuel Type"]

    data.dropna(subset=required_columns, inplace=True)

    # Step 2: Convert 'Activity %' column to numeric, setting errors='coerce' to turn invalid parsing into NaN
    data["Activity %"] = pd.to_numeric(data["Activity %"], errors='coerce')

    # Replace 0 values in 'Activity %' with 0.5
    data["Activity %"].replace(0, 0.5, inplace=True)

    # Ensure fuel consumption columns are numeric
    data['Fuel Cons t/day (Pre Application)'] = pd.to_numeric(data['Fuel Cons t/day (Pre Application)'],errors='coerce')

    data['Fuel Cons t/day (Post Application)'] = pd.to_numeric(data['Fuel Cons t/day (Post Application)'],errors='coerce')



    # Then, recheck for any missing data after conversion and handle them as before
    missing_data = data[relevant_columns].isnull()
    missing_rows = data[missing_data.any(axis=1)]

    # Print the count of these rows, if any
    if not missing_rows.empty and output:
        print(f'There are {len(missing_rows)} rows with missing values.')
        for index, row in missing_rows.iterrows():
            missing_cols = [col for col in relevant_columns if pd.isnull(row[col])]
            if missing_cols:  # Only print if there are missing columns
                print(f"Vessel Name: {row['Vessel Name']}")
                print("Missing columns: ", missing_cols)


    # Drop rows with missing values in relevant columns
    data = data[relevant_columns].dropna()

    vesselNames = list(data["Vessel Name"].unique())
    vesselFuelBurnNames = [name + " Fuel Burned Today (t)" for name in vesselNames]
    CO2SavingsNames = [name + " CO2 Saved Today (t)" for name in vesselNames]


    totalCO2 = 0
    totalVOC = 0
    totalCu = 0
    totalPaint = 0

    columnNames = ["Date", "Total VOC (kg)","Total Cu (kg)","Total Paint (kg)"]
    columnNames.extend(vesselFuelBurnNames)
    columnNames.extend(CO2SavingsNames)

    finalDataframe = pd.DataFrame(columns = columnNames)

    # Change string date to datetime object
    data['Date Coated'] = pd.to_datetime(data['Date Coated'])

    #calculates the environmental metrics for each of the days provided in the list
    for day in days:

        day_dt = pd.to_datetime(day)


        #hfo is default fuel type if nothig is provided
        data["Fuel Price"] = data["Fuel Type"].map(lambda x: fuelPrices.get(x, fuelPrices['HFO']))

        # Calculate days since application
        data['Days Since Coating'] = (day_dt - data['Date Coated']).dt.days.apply(lambda x: max(0,x))

        # Make sure that the fuel types exist in the emission factors dictionary
        data['Fuel Emission Factor'] = data['Fuel Type'].map(emission_factors)

        if data['Fuel Emission Factor'].isnull().any():
            print(
                "Warning: Some fuel types are not found in the emission factors. These will result in NaN values in the 'CO2 Prevented (t)' calculation.")
            print("Missing fuel types: ", data.loc[data['Fuel Emission Factor'].isnull(), 'Fuel Type'].unique())

        # Calculate CO2 prevented

        #add efficiency degradation over time
        #data['Fuel Cons t/day (Post Application)'] = data.apply(lambda row: adjusted_fuel_consumption(row['Days Since Coating'], row['Fuel Cons t/day (Pre Application)'], row['Fuel Cons t/day (Post Application)']), axis=1)

        data['Fuel Cons t/day (Post Application Modified)'] = data.apply(lambda row: adjusted_fuel_consumption(row['Days Since Coating'], row['Fuel Cons t/day (Pre Application)'], row['Fuel Cons t/day (Post Application)']), axis=1)

        data["Adjusted vs Fixed Fuel Rate"] = data['Fuel Cons t/day (Post Application)'] - data['Fuel Cons t/day (Post Application Modified)']



        # Calculate CO2 prevented today
        #data['CO2 Prevented Today (t)'] = (data['Fuel Cons t/day (Pre Application)'][-1] - data[
                                        #'Fuel Cons t/day (Post Application Modified)'][-1]) * data['Fuel Type'].map(emission_factors)

        # Check if coating type is "PROP" or if days since coating is zero (or less)
        cond = ((data["Coating Type"] == "PROP") | (data['Days Since Coating'] <= 0))

        dayscond = (data['Days Since Coating'] <= 0)

        # Calculate VOC avoided
        data['Total Application VOC Avoided (kg)'] = np.where(cond, 0, data['Coated Hull Area (m^2)'] * 0.44)

        # Calculate Paint Avoided
        data['Paint Avoided (kg)'] = np.where(cond, 0, data['Coated Hull Area (m^2)'] * 1.82)

        # Calculate Copper Prevented
        data['Copper Prevented (kg)'] = np.where(cond, 0, 14.49 * data['Days Since Coating'] / 1000 / 100 * data[
            'Coated Hull Area (m^2)'])

        # Calculate Silicone Oil Avoided
        data['Silicone Oil Avoided Today (kg)'] = np.where(cond, 0,   0.00001666 *data[ 'Coated Hull Area (m^2)'])

        # Calculate fuel savings in terms of $USD
        # It assumes that Fuel Cons t/day (Pre Application) and Fuel Cons t/day (Post Application) are in tons

        data['Fuel Savings Modified (t/day)'] = data['Fuel Cons t/day (Pre Application)'] - data[
            'Fuel Cons t/day (Post Application Modified)']

        #data['Fuel Savings Today ($USD)'] = np.where(dayscond,0,(data['Fuel Cons t/day (Pre Application)'] - data['Fuel Cons t/day (Post Application Modified)']) * data["Fuel Price"] * data["Activity %"])

        try:
            data['Fuel Savings Today ($USD)'] = np.where(dayscond, 0, (data['Fuel Cons t/day (Pre Application)'] - data[
                'Fuel Cons t/day (Post Application Modified)']) * data["Fuel Price"] * data["Activity %"])
        except TypeError:
            print("\nDebugging problematic multiplication:")
            print("Fuel Cons t/day (Pre Application):", data['Fuel Cons t/day (Pre Application)'])
            print("Fuel Cons t/day (Post Application Modified):", data['Fuel Cons t/day (Post Application Modified)'])
            print("Fuel Price:", data["Fuel Price"])
            print("Activity %:", data["Activity %"])
            raise  # This will re-raise the caught exception after printing the values.

        data["Fuel Savings Today (t)"] = np.where(dayscond,0,(data['Fuel Cons t/day (Pre Application)'] - data['Fuel Cons t/day (Post Application Modified)'])  * data["Activity %"])

        data["NOX Saved Today (kg)"] = data["Fuel Savings Today (t)"] * data["Fuel Type"].map(lambda x: NOXEmmision.get(x, NOXEmmision['HFO']))

        data["VOC Saved Today (kg)"] = data["Fuel Savings Today (t)"] * data["Fuel Type"].map(lambda x: VOCEmmision.get(x, VOCEmmision['HFO']))

        data["SOX Saved Today (kg)"] = data["Fuel Savings Today (t)"] * data["Fuel Type"].map(lambda x: SOXEmmision.get(x, SOXEmmision['HFO']))

        data["CO2 Saved Today (t)"] = np.where(dayscond,0,(data['Fuel Cons t/day (Pre Application)'] - data['Fuel Cons t/day (Post Application Modified)']) * data['Fuel Type'].map(emission_factors)) * data["Activity %"]

        data["Fuel Burned Today (t)"] = data['Fuel Cons t/day (Post Application Modified)'] * data["Activity %"]

        data["Copper Savings (5 year estimate) (kg)"] = np.where(cond,0,data['Coated Hull Area (m^2)'] / 1000 / 100 * 14.49 * 365 * 5 ) # add logic here



        totalApplicationVOC = sum(data['Total Application VOC Avoided (kg)'])
        totalCu = sum(data['Copper Prevented (kg)'])
        totalCu5Year = sum(data["Copper Savings (5 year estimate) (kg)"])
        totalPaint = sum(data['Paint Avoided (kg)'])
        totalFuelPostModified = sum(data['Fuel Cons t/day (Post Application Modified)'])
        totalFuelPost = sum(data['Fuel Cons t/day (Post Application)'])
        totalFuelPre = sum(data['Fuel Cons t/day (Pre Application)'])
        diffFuel = totalFuelPost - totalFuelPostModified
        totalCO2SavingsToday = sum(data["CO2 Saved Today (t)"])
        fuelSavingsToday = sum(data["Fuel Savings Today (t)"])
        totalSiliconAvoidedToday = sum(data["Silicone Oil Avoided Today (kg)"])

        totalVOCSavedToday = sum(data["VOC Saved Today (kg)"])
        totalNOXSavedToday = sum(data["NOX Saved Today (kg)"])
        totalSOXSavedToday = sum(data["SOX Saved Today (kg)"])


        new_row_data = {
            "Date": day,
            "Total CO2 Savings Today (tons)" : totalCO2SavingsToday,
            "Total Application VOC (kg)": totalApplicationVOC,
            "Total Cu (kg)": totalCu,
            "Total Cu (5 Year Estimate) (kg)" : totalCu5Year,
            "Total Paint (kg)": totalPaint,
            "Total Fuel Post Modified (tons)": totalFuelPostModified,
            "Total Fuel Post (tons)": totalFuelPost,
            "Total Fuel Pre (tons)": totalFuelPre,
            "Fuel Difference" : diffFuel,
            "Fuel Savings Today (t)" : fuelSavingsToday,
            "Total FUEL VOC Saved Today (kg)" : totalVOCSavedToday,
            "Total NOX Saved Today (kg)": totalNOXSavedToday,
            "Total SOX Saved Today (kg)": totalSOXSavedToday,
            "Total Silicon Avoided Today (kg)": totalSiliconAvoidedToday
        }


        fleetDailyFuelRate = []
        fleetDailyMoneySaved = []
        fleetDailyCO2Saved = []

        # Iterate through the vessel names
        numvessels = len(data['Vessel Name'].unique())

        for vessel in data['Vessel Name'].unique():
            # Note: vessel should be in the format "VesselName Cumulative Fuel Savings ($USD)"
            # Get the fuel savings for this vessel on this date

            thisEconomicSavings = data.loc[data['Vessel Name'] == vessel, 'Fuel Savings Today ($USD)'].values[0]
            thisFuelBurn = data.loc[data['Vessel Name'] == vessel, 'Fuel Burned Today (t)'].values[0]
            thisCO2Saved = data.loc[data['Vessel Name'] == vessel, 'CO2 Saved Today (t)'].values[0]

            fleetDailyFuelRate.append(thisFuelBurn)
            fleetDailyMoneySaved.append(thisEconomicSavings)
            fleetDailyCO2Saved.append(thisCO2Saved)

            # Add it to the new row data under the correct column
            new_row_data[vessel + " Money Saved Today ($USD)"] = thisEconomicSavings
            new_row_data[vessel + " Fuel Burned Today (t)"] = thisFuelBurn
            new_row_data[vessel + " CO2 Saved Today (t)"] = thisCO2Saved

        new_row_data["Total Daily Fuel Burn (t)"] = sum(fleetDailyFuelRate)
        new_row_data["Total Daily Money Saved ($USD)"] = sum(fleetDailyMoneySaved)
        new_row_data["Total CO2 Saved Today (t)"] = sum(fleetDailyCO2Saved)


        new_row = pd.DataFrame([new_row_data])

        if output:
            for vessel in data['Vessel Name'].unique():
                vessel_data = data[data['Vessel Name'] == vessel]
                print(f"\nVessel Name: {vessel}")
                print(f"Fuel Type Used: {vessel_data['Fuel Type'].iloc[0]}")
                print(f"Total CO2 (tons): {vessel_data['CO2 Prevented (t)'].sum()}")
                print(f"Total VOC (kg): {vessel_data['VOC Avoided (kg)'].sum()}")
                print(f"Total Cu (kg): {vessel_data['Copper Prevented (kg)'].sum()}")
                print(f"Total Paint (kg): {vessel_data['Paint Avoided (kg)'].sum()}")


        if output : print(f"Merging {day}")
        finalDataframe = pd.concat([finalDataframe, new_row], ignore_index=True)

        finalDataframe['Cumulative CO2 Prevented (tons)'] = finalDataframe["Total CO2 Savings Today (tons)"].cumsum()
        finalDataframe["Cumulative NOX Prevented (kg)"] = finalDataframe["Total NOX Saved Today (kg)"].cumsum()
        finalDataframe["Cumulative SOX Prevented (kg)"] = finalDataframe["Total SOX Saved Today (kg)"].cumsum()
        finalDataframe["Cumulative VOC Prevented (kg)"] = finalDataframe["Total FUEL VOC Saved Today (kg)"].cumsum()
        finalDataframe["Cumulative Avoided NOX and SOX (t)"] = (finalDataframe["Cumulative SOX Prevented (kg)"] + finalDataframe["Cumulative NOX Prevented (kg)"]) / 1000
        finalDataframe["Cumulative Silicone Oil Prevented (kg)"] = finalDataframe["Total Silicon Avoided Today (kg)"].cumsum()

        finalDataframe["TOTAL VOC SAVED (kg)"] = finalDataframe["Cumulative VOC Prevented (kg)"] + finalDataframe["Total Application VOC (kg)"]

        finalDataframe["Carbon Credit Savings (USD)"] = finalDataframe['Cumulative CO2 Prevented (tons)'] * ETSprice


        # merge the cumulative_savings dataframe with the finalDataframe

        #if day == '2023-10-07':
            #data.to_excel("data snapshot.xlsx")


        if output : print(f"----------------------------\nEnvironmental Metrics on {day}\nTotal CO2 (tons) {totalCO2}\nTotal VOC (kg) {totalVOC}\nTotal Cu (kg) {totalCu}\nTotal Paint (kg) {totalPaint} ")

    #finalDataframe["Fleet Fuel Usage (t)"] =

    finalDataframe.to_excel(f"results/Results {days[0]}-{days[-1]} V{numvessels}.xlsx")

    if debug:
        print(f"There are {len(data['Vessel Name'].unique())} unique vessels.")
        print(f"Unique Vessel Names: {data['Vessel Name'].unique()}")


    return finalDataframe


def calculate_cumulative_savings(df, date, vessel):
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create the column names based on the vessel name
    money_saved_column = f"{vessel} Money Saved Today ($USD)"
    co2_saved_column = f"{vessel} CO2 Saved Today (t)"

    # Filter the dataframe to include only dates up to and including the specified date
    df_filtered = df[df['Date'] <= date]

    # Calculate the cumulative total CO2 saved and money saved for the given vessel up to and including the given date
    total_co2_saved = df_filtered[co2_saved_column].sum()
    total_money_saved = df_filtered[money_saved_column].sum()

    return total_co2_saved, total_money_saved


def plot_return_on_investmentTWOBARS(df, date, vessel, application_cost):
    # Calculate the cumulative savings
    df['Date'] = pd.to_datetime(df['Date'])


    _, total_money_saved = calculate_cumulative_savings(df, date, vessel)

    # Calculate the return on investment
    roi = (total_money_saved / application_cost) * 100

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Application Cost ($USD)', 'Total Money Saved ($USD)'], [application_cost, total_money_saved],
            color=['red', 'green'])
    #plt.title(f'Estimated Return on Investment for {vessel} up to {date}\nReturn on Investment: {roi:.2f}%')
    plt.title(f'Estimated Return on Investment for Ro-Ro Ferry up to {date}\nReturn on Investment: {roi:.2f}%')

    plt.ylabel('Amount ($USD)')
    plt.text(0, application_cost / 2, f'Application Cost: ${application_cost:.2f}', fontsize=12, ha='center')
    plt.text(1, total_money_saved / 2, f'Total Savings: ${total_money_saved:.2f}', fontsize=12, ha='center')
    plt.grid(True)
    plt.show()


def plot_return_on_investmentTIMELINE(df, date, vessel, application_cost):
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create the column names based on the vessel name
    money_saved_column = f"{vessel} Money Saved Today ($USD)"
    co2_saved_column = f"{vessel} CO2 Saved Today (t)"

    # Find the date where the "FINNSKY Money Saved Today ($USD)" column first becomes non-zero
    start_date = df[df[money_saved_column] > 0]['Date'].min()

    # Filter the dataframe to include only dates from the start_date to the specified date
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= date)].copy()

    # Calculate the cumulative total CO2 saved and money saved for the given vessel up to and including the given date
    df_filtered['cumulative_money_saved'] = df_filtered[money_saved_column].cumsum()

    # Calculate net savings
    df_filtered['net_savings'] = df_filtered['cumulative_money_saved'] - application_cost

    # Calculate the return on investment
    total_money_saved = df_filtered['cumulative_money_saved'].iloc[-1]
    roi = (total_money_saved / application_cost) * 100

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group the data by week to reduce the number of bars in the plot
    weekly_data = df_filtered.resample('W', on='Date')['net_savings'].last()

    # Increase the bar width to make the bars a bit thicker
    ax.bar(weekly_data.index, weekly_data,
           color=np.where(weekly_data >= 0, 'g', 'r'), width=7.0)

    ax.set_title(f'Estimated Return on Investment for Ro-Ro Ferry up to {date}\nReturn on Investment: {roi:.2f}%')
    ax.set_ylabel('Net Savings ($USD)')
    ax.axhline(0, color='black', linewidth=0.5)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    ax.grid(True)
    plt.show()
# Test the function with a sample date, vessel, and application cost


def plot_environmental_impact(df, date, vessel, amount_Cu, amount_VOC):
    # Calculate the cumulative savings
    total_co2_saved, _ = calculate_cumulative_savings(df, date, vessel)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Total CO2 Saved (t)', 'Total Cu (kg)', 'Total VOC (kg)'], [total_co2_saved, amount_Cu, amount_VOC],
            color=['green', 'blue', 'red'])
    plt.title(f'Environmental Impact for Ro-Ro Ferry up to {date}')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.show()


def fleetEnvImpactOnDate(data, date_string, debug=False):
    """
    Visualize the metrics 'CO2 Prevented (t)', 'Total VOC (kg)', 'Total Cu (kg)', and 'Total Paint (kg)'
    on a specific date.

    Parameters:
    - data: A pandas DataFrame that includes the metrics and a 'Date' column.
    - date_string: A string representing the date of interest, in the format 'YYYY-MM-DD'.
    - debug: Boolean to turn on debugging print statements.
    """

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Convert the date_string to a datetime object
    date_of_interest = pd.to_datetime(date_string)

    # Filter the data for the date of interest
    date_data = data[data['Date'] == date_of_interest]

    if debug:
        print("Data for the date of interest:", date_data)

    # Create a bar plot for each of the metrics
    metrics = ['Cumulative CO2 Prevented (tons)', 'Cumulative Avoided NOX and SOX (t)',
               'Total Cu (5 Year Estimate) (kg)', 'Total Paint (kg)']
    units = ['tons', 't', 'kg', 'kg']
    values = date_data[metrics].values.flatten()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    for i, bar in enumerate(bars):
        value = round(values[i])
        unit = units[i]
        text = f"{value} {unit}"
        text_x = bar.get_x() + bar.get_width() / 2.0
        text_y = bar.get_height() + 0.05
        plt.text(text_x, text_y, text, ha='center', va='bottom', fontsize=14)

    plt.xlabel('GIT Environmental Impact')
    plt.ylabel('Value')
    plt.title(f'Metrics for {date_string}')
    plt.xticks(rotation=45)
    plt.show()


def movingPlot(df):
    # Identify the columns that represent individual vessel's daily CO2 savings
    vessel_columns = [col for col in df.columns if "CO2 Saved Today (t)" in col and "Total" not in col]
    vessel_names = [col.split(" CO2")[0] for col in vessel_columns]

    # Create the base line for total CO2 saved
    fig = go.Figure(data=[go.Scatter(x=df['Date'], y=df['Cumulative CO2 Prevented (tons)'], mode='lines', name='Total CO2 Saved')])

    # Create frames for each date
    frames=[]
    saving_vessels = {}

    for k in range(1, df['Date'].size):
        frame_data = [go.Scatter(x=df['Date'][:k+1], y=df['Cumulative CO2 Prevented (tons)'][:k+1], mode='lines', name='Total CO2 Saved')]
        annotations = []

        for i, (vessel, vessel_col) in enumerate(zip(vessel_names, vessel_columns)):
            vessel_data = df[vessel_col][:k+1]

            if vessel_data.iloc[-1] > 0 and vessel_data.iloc[-2] == 0:
                saving_vessels[vessel] = (df['Date'].iloc[k], df['Cumulative CO2 Prevented (tons)'].iloc[k])

            if vessel in saving_vessels:
                annotations.append(dict(x=saving_vessels[vessel][0], y=saving_vessels[vessel][1],
                                        xref="x", yref="y",
                                        text=vessel,
                                        showarrow=True, arrowhead=7, ax=0, ay=-50 - i*20)) # adjust arrow length to prevent overlapping

            frame_data.append(go.Scatter(x=df['Date'][:k+1], y=vessel_data, mode='lines', name=vessel+' CO2 Saved'))

        frames.append(go.Frame(data=frame_data, layout=go.Layout(annotations=annotations, xaxis=dict(range=[df['Date'].min(), df['Date'].iloc[k]]))))

    fig.frames = frames

    fig.update_layout(
        title="Total CO2 Saved over time (tons)",
        yaxis=dict(range=[0, df['Cumulative CO2 Prevented (tons)'].max()], autorange=False, tickformat=','),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300,"easing": "quadratic-in-out"}}])])])

    fig.show()


def create_bar_chart_race(df, start_date=None, end_date=None):

    # Use provided start and end dates, or default to first and last dates in DataFrame
    if start_date is None:
        start_date = df['Date'].min()
    if end_date is None:
        end_date = df['Date'].max()

    df = df[df["Date"] >= start_date]
    df = df[df["Date"] <= end_date]


    df = process_dataframe(df)

    # Set 'Date' as the index of the DataFrame
    df.set_index('Date', inplace=True)

    # Generate the bar chart race
    bcr.bar_chart_race(
        df=df,
        filename='co2_savings.mp4',  # Save the animation as 'co2_savings.mp4'
        orientation='h',
        sort='desc',
        n_bars=6,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=2,
        interpolate_period=False,
        label_bars=True,
        bar_size=.95,
        period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
        period_summary_func=lambda v, r: {'x': .99, 'y': .18,
                                          's': f'Total CO2 saved: {v.nlargest(6).sum():,.0f}',
                                          'ha': 'right', 'size': 8, 'family': 'Courier New'},
        dpi=144,
        cmap='dark12',
        title='CO2 Saved by Vessel Over Time',
        bar_label_size=7,
        tick_label_size=7,
        shared_fontdict={'family' : 'DejaVu Sans', 'color' : '.1'},
        scale='linear',
        writer=None,
        fig=None,
        bar_kwargs={'alpha': .7},
        filter_column_colors=False)

    # Show the figure
    plt.show()# Call the function with the DataFrame


# Function to create the enhanced area plot
# Updated function to include start_date and end_date parameters
def plot_enhanced_area(ax, df_savings, start_date, end_date, debug=False):
    # Filter the DataFrame based on the provided start_date and end_date
    df_savings_filtered = df_savings[(df_savings.index >= start_date) & (df_savings.index <= end_date)]

    # Calculate the cumulative sum of the savings
    df_savings_filtered['Cumulative Value ($USD)'] = df_savings_filtered['Total Daily Money Saved ($USD)'].cumsum()

    if debug:
        print("Filtered Data:")
        print(df_savings_filtered)

    # Create a colormap for the gradient shading
    cmap = LinearSegmentedColormap.from_list('my_cmap', ['#0D1631', '#87D057'], N=100)

    # Plot the data as an area plot with gradient shading for cumulative savings
    ax.fill_between(df_savings_filtered.index,
                    df_savings_filtered['Cumulative Value ($USD)'],
                    color='skyblue',
                    alpha=0.4)

    ax.plot(df_savings_filtered.index,
            df_savings_filtered['Cumulative Value ($USD)'],
            color='Slateblue',
            alpha=0.6)

    # Adding gradient color based on cumulative savings
    for i in range(1, len(df_savings_filtered)):
        ax.fill_betweenx([0, df_savings_filtered['Cumulative Value ($USD)'].iloc[i]],
                         df_savings_filtered.index[i - 1],
                         df_savings_filtered.index[i],
                         color=cmap(df_savings_filtered['Cumulative Value ($USD)'].iloc[i] / df_savings_filtered[
                             'Cumulative Value ($USD)'].max()))

    # Add a colorbar to explain the gradient color
    norm = plt.Normalize(df_savings_filtered['Cumulative Value ($USD)'].min(),
                         df_savings_filtered['Cumulative Value ($USD)'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')

    # Format colorbar ticks to currency style
    formatter = ticker.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x))
    cbar.ax.yaxis.set_major_formatter(formatter)

    cbar.ax.yaxis.label.set_color('white')
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color('white')

    if debug:
        print("Plotting completed.")


def dashboard(data, tick_size=40, text_size=20, background_image_path=None):
    # Add the main and secondary color codes here
    main_colors = ['#1B444E', '#32CCC8', '#87D057', '#FFFFFF', '#000000']
    secondary_colors = ['#D54950', '#0D1631']

    legendSize = 15

    fig = plt.figure(figsize=(11, 8.5))  # standard 8.5 x 11 sheet in landscape

    # If a background image path is provided, load the image and set it as the figure background
    # If a background image path is provided, load the image and set it as the figure background
    if background_image_path:
        img = Image.open(background_image_path)
        img = np.array(img)

        # Create an axis that covers the whole figure
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.imshow(img, aspect='auto')
        ax.axis('off')  # Hide the axis

    # Create grid layout
    gs = fig.add_gridspec(4, 5)  # Adjusting to 5 columns to accommodate your pies

    # Adding subplots for goalVSactual
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 4])


    # Plot the goalVSactual on these axes
    goalVSactual(data, ax_list=[ax0, ax1, ax2, ax3, ax4], show=False,fontScale=75)

    # Placeholder for other plots
    ax5 = fig.add_subplot(gs[1:, 2:5])
    ax6 = fig.add_subplot(gs[1:2, 0:2])
    ax7 = fig.add_subplot(gs[2:4, 0:2])

    # Assuming 'Date' is the column in your DataFrame representing time
    end_date = datetime.now()

    #start date should be the first day of the current year
    start_date = datetime(end_date.year, 1, 1)

    print(f"End Date is {end_date}")
    print(f"Start Date is {start_date}")

    #start_date = end_date - timedelta(days=12 * 30)  # Approximating a month as 30 days

    if "Total CO2 Savings Today (tons)" in data.columns:
        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        ax5.plot(filtered_data['Date'], filtered_data["Total CO2 Savings Today (tons)"],color = main_colors[0],label = "Daily CO2 Avoided (tons)")

        # Formatting the x-axis with month names
        ax5.xaxis.set_major_locator(mdates.MonthLocator())  # Setting major locator to month
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Formatting the x-axis ticks to show month names
        ax5.set_xlabel("")
        ax5.yaxis.set_label_position("right")
        ax5.set_ylabel("C02 Saved Per Day",fontsize = 20* text_size)
        ax5.grid()
        ax5.set_title("")
        ax5.legend(fontsize = legendSize)
    else:
        ax5.text(0.5, 0.5, "Data for 'Total CO2 Savings Today (tons)' not found in DataFrame",
                 ha='center', va='center', color='red', fontsize=12* text_size)

    # Implement Plot 3 with dual y-axes
    if "Cumulative Avoided NOX and SOX (t)" in data.columns and "Total Cu (5 Year Estimate) (kg)" in data.columns:
        NoxSoxcolor = main_colors[1]
        CuColor = secondary_colors[0]


        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

        ax7.plot(filtered_data['Date'], filtered_data["Cumulative Avoided NOX and SOX (t)"], color=NoxSoxcolor,
                 label='NOX and SOX Avoided (t)')
        #ax7.set_ylabel("NOX and SOX Avoided (t)", color=NoxSoxcolor,fontsize = 20* text_size)
        ax7.tick_params(axis='y', labelcolor=NoxSoxcolor)

        ax7_twin = ax7.twinx()  # instantiate a second axes that shares the same x-axis

        ax7_twin.plot(filtered_data['Date'], filtered_data["Total Cu (5 Year Estimate) (kg)"], color=CuColor,
                      label='Total Cu Avoided (kg)')
        #ax7_twin.set_ylabel("Cu Avoided (kg)", color=CuColor,fontsize = 20* text_size)
        ax7_twin.tick_params(axis='y', labelcolor="white")

        ax7.xaxis.set_major_locator(mdates.MonthLocator())  # Setting major locator to month
        ax7.xaxis.set_major_formatter(
            mdates.DateFormatter('%b'))  # Formatting the x-axis ticks to show full month names

        ax7.legend(fontsize=legendSize * 0.75, loc='upper left', bbox_to_anchor=(0, 1))
        ax7_twin.legend(fontsize=legendSize * 0.75, loc='upper left', bbox_to_anchor=(0, 0.90))  # Adjust 0.92 as needed
        ax7.grid()

        ax7.set_title("Cumulative Avoided NOX and SOX (t) and Total Cu (5 Year Estimate) (kg)")
    else:
        ax7.text(0.5, 0.5, "Data for the specified columns not found in DataFrame", ha='center', va='center',
                 color='red', fontsize=12* text_size)

    # Extract the Date and 'Total Daily Money Saved ($USD)' columns
    data['Date'] = pd.to_datetime(data['Date'])
    df_savings = data[['Date', 'Total Daily Money Saved ($USD)']]

    # Drop any NA values and set Date as index for easier plotting
    df_savings = df_savings.dropna()
    df_savings.set_index('Date', inplace=True)

    # Call the function to plot the enhanced area plot on ax6
    plot_enhanced_area(ax6, df_savings,start_date,end_date)

    ax6.xaxis.set_major_locator(mdates.MonthLocator())  # Setting major locator to month
    ax6.xaxis.set_major_formatter(
        mdates.DateFormatter('%b'))  # Formatting the x-axis ticks to show full month names

    #get rid of the y ticks and markers on ax6
    ax6.set_yticks([])
    ax6.set_yticklabels([])
    ax6.set_ylabel("Customer Savings\n($USD)",fontsize = 20* text_size,color = "white")

    ax6.set_title("")
    ax7.set_title("")

    # Adjusting the x-axis of ax5
    ax5.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Displaying ticks for Jan, Apr, Jul, Oct
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Displaying in the format "Aug, 2023"

    # Adjusting the x-axis of ax6
    ax6.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Displaying ticks for Jan, Apr, Jul, Oct
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Displaying in the format "Aug, 2023"

    # Adjusting the x-axis of ax7
    ax7.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Displaying ticks for Jan, Apr, Jul, Oct
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Displaying in the format "Aug, 2023"

    # Adjust tick size for all subplots
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,ax7_twin]:
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)

    # Add this line where you have looped through all axes to adjust tick sizes
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax7_twin]:
        set_axis_font(ax, font_properties)  # Call our helper function to set font properties

    for ax in [ax5, ax6, ax7]:
        # Change the color of x-axis ticks to white
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_color('white')

        # Change the color of y-axis ticks to white
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_color('white')

    """
    implement these metrics
    top=0.895,
    bottom=0.048,
    left=0.048,
    right=0.942,
    hspace=0.408,
    wspace=0.965
    """

    # Adjusting the spacing between subplots
    fig.subplots_adjust(top=0.895, bottom=0.078, left=0.043, right=0.967, hspace=0.408, wspace=0.965)


    #plt.tight_layout()
    plt.savefig("Dashboards/dash " + background_image_path , dpi=300)
    plt.show()

def process_dataframe(df):
    # Identify columns that match the format "STOLT STREAM CO2 Saved Today (t)"
    co2_cols = [col for col in df.columns if 'CO2 Saved Today (t)' in col and 'Total CO2 Saved Today (t)' not in col]

    # Initialize new dataframe with date column
    new_df = df[['Date']].copy()

    # Process each CO2 column
    for col in co2_cols:
        # Generate new column name
        new_col_name = col.replace(' Saved Today (t)', ' Total Saved (t)')

        # Compute cumulative sum and add to new dataframe
        new_df[new_col_name] = df[col].cumsum()

    return new_df


def vessel_annotated_metrics(df, column_to_plot, c02Lim=1):
    """
    Plot a given column along with annotations for major vessel contributions.

    Parameters:
    - df (DataFrame): The data frame containing the data.
    - column_to_plot (str): The name of the column to plot.
    - c02Lim (float): The threshold for annotating significant CO2 savings. Default is 1.
    """
    # Convert the 'Date' column to datetime format if it's not already
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the specified column as the main graph
    ax.plot(df['Date'], df[column_to_plot])
    ax.set_title(f'{column_to_plot} with Major Contributions')
    ax.set_xlabel('Date')
    ax.set_ylabel(column_to_plot)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True)

    # Find columns in the format "{Vessel Name} CO2 Saved Today (t)"
    vessel_columns = [col for col in df.columns if 'CO2 Saved Today (t)' in col]

    vessel_columns.remove("Total CO2 Saved Today (t)")

    # Variable to alternate the side where the annotation appears
    alternate_side = True

    try:
        # Loop through each vessel column to find the first non-zero value
        for vessel_col in vessel_columns:
            first_non_zero_row = df[df[vessel_col] > 0].iloc[0]

            # Check if the first non-zero value is greater than the specified limit (c02Lim)
            if first_non_zero_row[vessel_col] > c02Lim:
                date_to_annotate = first_non_zero_row['Date']
                value_to_annotate = first_non_zero_row[column_to_plot]
                vessel_name = vessel_col.split(' CO2 Saved Today (t)')[0]

                # Decide the side for annotation based on the alternate_side flag
                offset_value = 1500 if alternate_side else -1500

                # Annotate the plot with larger font size and longer arrow
                ax.annotate(vessel_name,
                            xy=(date_to_annotate, value_to_annotate),
                            xytext=(date_to_annotate, value_to_annotate + offset_value),
                            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                            fontsize=12)

                # Toggle the alternate_side flag for next iteration
                alternate_side = not alternate_side

    except Exception as e:
        if debug:
            print(f"An error occurred while annotating the plot: {e}")

    # Format the date labels to prevent overlapping
    fig.autofmt_xdate()

    # Show the final plot
    plt.tight_layout()
    plt.show()


def realtimeVS5yr(data, valsOnDate="2023-10-03", debug=False):
    """
    Plot the trends for "Total Cu (5 Year Estimate) (kg)" and "Total Cu (kg)" against the "Date" column.
    Optionally, add a vertical line and annotations for values on a specific date.

    Parameters:
    - data: A pandas DataFrame that includes the 'Date', 'Total Cu (5 Year Estimate) (kg)', and 'Total Cu (kg)' columns.
    - valsOnDate: Optional date string in the form 'YYYY-MM-DD' to show a vertical line and annotations.
    - debug: Boolean to turn on debugging print statements.
    """

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    if debug:
        print("Data head:", data.head())

    plt.figure(figsize=(14, 8))

    plt.plot(data['Date'], data['Total Cu (5 Year Estimate) (kg)'], label='Total Cu (5 Year Estimate) (kg)', color='b')
    plt.plot(data['Date'], data['Total Cu (kg)'], label='Total Cu (kg)', color='r')

    if valsOnDate:
        date = pd.to_datetime(valsOnDate)
        cu_5yr = data.loc[data['Date'] == date, 'Total Cu (5 Year Estimate) (kg)'].values[0]
        cu_realtime = data.loc[data['Date'] == date, 'Total Cu (kg)'].values[0]

        plt.axvline(x=date, color='g', linestyle='--')
        #increase the size of the annotation
        plt.annotate(f"{valsOnDate}\n5-yr Cu: {round(cu_5yr,1)} kg\nRealtime Cu: {round(cu_realtime,1)} kg", (date, cu_5yr),
                     textcoords="offset points", xytext=(0, 10), ha='center',fontsize = 17)

    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Cu Savings (kg)', fontsize=20)
    plt.title('GIT Sustainable Coatings\n5 Year Vs Realtime Cu savings', fontsize=25)

    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)

    plt.show()


def goalVSactual(data, ax_list=None, target_CO2=48000, target_VOC=72000, target_NOX_SOX=1500, target_Cu=31500, target_Si = 360, debug=True, show = True,fontScale = 1):
    """
    Plot projected savings as pie charts for different metrics.
    - data: DataFrame containing the metrics columns.
    - target_CO2, target_VOC, target_NOX_SOX, target_Cu: target values for the metrics.
    - debug: Boolean to turn on debugging print statements.
    """

    # Convert 'Date' to datetime and filter data for 2023 up to today's date
    data['Date'] = pd.to_datetime(data['Date'])


    #startingCopper = data['Total Cu (5 Year Estimate) (kg)'].iloc[0]
    startingCopper = data[data['Date'] == '2024-01-01']['Total Cu (5 Year Estimate) (kg)'].iloc[0]

    today = datetime.now() - timedelta(days=1)
    #today = datetime.now()
    data_2023 = data[(data['Date'].dt.year == 2024) & (data['Date'] <= today)]



    if debug:
        print("Data for 2023 up to today:", data_2023.head())

    # Calculate the current values for the metrics
    start_values = data_2023.iloc[0]
    current_values = data_2023.iloc[-1]

    current_CO2 = current_values['Cumulative CO2 Prevented (tons)'] - start_values['Cumulative CO2 Prevented (tons)']
    current_VOC = current_values['TOTAL VOC SAVED (kg)'] - start_values['TOTAL VOC SAVED (kg)']
    current_NOX_SOX = current_values['Cumulative Avoided NOX and SOX (t)'] - start_values[
        'Cumulative Avoided NOX and SOX (t)']
    #current_Cu = current_values['Total Cu (5 Year Estimate) (kg)'] - start_values['Total Cu (5 Year Estimate) (kg)']
    current_Cu = current_values['Total Cu (5 Year Estimate) (kg)'] - startingCopper
    current_Si = current_values["Cumulative Silicone Oil Prevented (kg)"] - start_values[
        'Cumulative Silicone Oil Prevented (kg)']

    #print out values for silicon oil
    print(f"Current Silicon Oil: {current_Si}")
    print(f"Target Silicon Oil: {target_Si}")

    targetList = [target_CO2, target_VOC, target_NOX_SOX, target_Cu, target_Si]

    metrics = [current_CO2, current_VOC, current_NOX_SOX, current_Cu,current_Si]
    targets = [target_CO2, target_VOC, target_NOX_SOX, target_Cu,target_Si]
    titles = [
        "Avoided Carbon Dioxide\n Equivalent (tons)",
        "Avoided\n Volatile Organic\nCompounds (kg)",
        "Avoided Sulfur &\n Nitrogen Oxides (tons)",
        "Avoided Copper from \nBiocides 5-Yr (kg)",
        "Avoided Silcone Oils (kg)"
    ]

    if ax_list is None:
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    else:
        axs = ax_list

    for i, ax in enumerate(axs):
        percentage_completion = (metrics[i] / targets[i]) * 100  # Calculate the percentage early for logic

        #in the case that we are beasts
        """    if percentage_completion > 100:
            percentage_completion = 100"""

        if debug:
            print(f"Metric: {metrics[i]}")
            print(f"Target: {targets[i]}")
            print(f"Percentage Completion: {percentage_completion}")
            print(f"Targets - Metrics: {targets[i] - metrics[i]}")

        difference = targets[i] - metrics[i]

        #this means we completed our goal
        if difference < 0 :
            difference = 0

        ax.pie([metrics[i], difference], labels=None, autopct=None, startangle=90,
               counterclock=False, wedgeprops=dict(width=0.3), colors=['#4CAF50', '#FFC107'])


        """# Handle the 100% case by drawing a full circle
        if percentage_completion >= 100 or metrics[i] == targets[i]:
            full_circle = plt.Circle((0, 0), 0.3, color='#4CAF50', fill=True)
            ax.add_artist(full_circle)
        else:  # The normal pie chart logic"""


        ax.set(aspect="equal")
        plt.sca(ax)
        plt.title(titles[i], fontsize=22 * fontScale)

        # Format the number into the format of 4,000 from 4000
        formatted_metric = "{:,}".format(round(metrics[i]))

        # Display the formatted metric and percentage of completion
        plt.text(0, 0.1, formatted_metric, ha='center', va='center', fontsize=30*fontScale)
        plt.text(0, -1.4, f"{percentage_completion:.2f}% Complete\nGoal : {targetList[i]}", ha='center', va='center', fontsize=25*fontScale)

    plt.savefig("goalsvsactual.png")
    plt.tight_layout()
    if show and ax_list is None:
        plt.show()
    return axs

#hsfo_380_prices = [480, 490, None, 485, 510, 534, 515, 530, 500, 600, 530, None, None, None, 580, None, 420, 485, 465, 485, None, 520, 430, 460, 445, 555, 515, None, None, 465, 570, None]


#prices = [price for price in hsfo_380_prices if price is not None]
#average_price = sum(prices) / len(prices)

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
# start date
start_date = date(2023, 1, 1)

# end date
end_date = date(2025, 1, 1)

# calculate the number of days in 2023
delta = end_date - start_date

# generate the list of days
dates = [(start_date + timedelta(days=i)).isoformat() for i in range(delta.days + 1)]
#jan to june end YTD

#def environmental_impact_calculation(data, emission_factors, days, fuelPrices,VOCEmmision,SOXEmmision,NOXEmmision, output,report = True):


#dates = ['2023-12-30','2023-12-31']
#results = environmental_impact_calculation(data, emission_factors, dates,fuelPrices,VOCEmmision,SOXEmmision,NOXEmmision,False,report = True,debug = True)
# Test the function with the given DataFrame, for Q2 and June




#print(results)

#plt.plot(results["Date"],results["CO2 Prevented (t)"])
#plt.show()



df = pd.read_excel(r"C:\Users\jmurp\Graphite Innovation & Technologies\Corporate - Documents\Post-Sales Operations\12-Energy & Efficiency\1 - Performance & Analysis\22 - Environmental Impact\EI Model\results\Results 2023-01-01-2025-01-01 V111.xlsx")

"""q_string, m_string, w_string, ytd_string = weeklyReport(df, quarter = 4,month = 10,week = 2)


print(q_string)
print(m_string)
print(ytd_string)
print(w_string)"""

#q_string, m_string, ytd_string = weeklyReport(df, quarter=2, month=6)

#df100Activity = pd.read_excel(r"C:\Users\jmurp\Graphite Innovation & Technologies\Corporate - Documents\Business Development\12-Energy & Efficiency\1 - Performance & Analysis\22 - Environmental Impact\EI Model\results\Results 2022-01-01-2026-03-01 V76.xlsx")
#goalVSactual(df,show = True)
#dfregActivity = pd.read_excel(r"C:\Users\jmurp\Graphite Innovation & Technologies\Corporate - Documents\Business Development\12-Energy & Efficiency\1 - Performance & Analysis\22 - Environmental Impact\EI Model\results\Results 2022-01-01-2025-01-01 V76.xlsx")
#realtimeVS5yr(df100Activity)
#plot_total_CO2_savings(df100Activity)
#plot_total_money_savings(df100Activity)
#fleetEnvImpactOnDate(df100Activity, "2023-10-03")
#dashboard(df,background_image_path="background1.png")
#dashboard(df,background_image_path="background2.png")
#dashboard(df,background_image_path="background3.png")
#dashboard(df,background_image_path="background4.png")
dashboard(df,background_image_path="background4.png")



#plot_total_CO2_savings(df)
#plot_total_money_savings(df)
#plot_environmental_impact(df, '2023-07-12', 'FINNSKY', 225.7, 2211)
#calculate_monthly_impact(df, 2023)
#movingPlot(df)




"""#testing for adjusted fuel consumption
# Test values
fuel_consumption_pre_application = 100  # Just an example
fuel_consumption_post_application = 90  # Just an example

# Create a list to store the fuel consumption values over 20 years (or 7300 days)
fuel_consumption_over_time = []

# For each day in the 20-year period
for days_since_application in range(7300):
    # Calculate the adjusted fuel consumption for that day
    fuel_consumption = adjusted_fuel_consumption(days_since_application, fuel_consumption_pre_application, fuel_consumption_post_application)
    # Add the value to the list
    fuel_consumption_over_time.append(fuel_consumption)

# Plot the fuel consumption values over time
plt.plot(fuel_consumption_over_time)
plt.title('Fuel Consumption Over Time')
plt.xlabel('Days Since Application')
plt.ylabel('Fuel Consumption')
plt.show()"""