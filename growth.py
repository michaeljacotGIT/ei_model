

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def growth_plot(df, date_column_name, title='Cumulative Sales Plot', xlabel='Date', ylabel='Cumulative Sales'):
    """
    Plots the cumulative sales given a dataframe with a single date column.

    The function creates a new DataFrame with every day from the first date in the list to the current date,
    counts the occurrences of each date in the original data, and then plots the cumulative sum.

    Parameters:
    df (DataFrame): DataFrame containing a single date column.
    date_column_name (str): The name of the date column in df.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    # Generate a DataFrame with dates from the first date to the current date
    start_date = df[date_column_name].min()
    #get the end of 2024
    end_date = pd.Timestamp(year=2024, month=12, day=31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_df = pd.DataFrame({'Date': all_dates})

    # Count occurrences of each date in the original data
    sales_counts = df[date_column_name].value_counts().rename('Sales')

    # Merge the counts with the all_dates DataFrame and fill missing values with 0
    merged_df = date_df.merge(sales_counts, how='left', left_on='Date', right_index=True).fillna(0)

    # Calculate the cumulative sum
    merged_df['Cumulative Sales'] = merged_df['Sales'].cumsum()

    #make a column that counts beside the date colums as 1 2 3 4 5 6 7 8 9 10
    merged_df["Day"] = np.arange(1,len(merged_df)+1)

    def f(x,a,b,c,d):
        #return a*np.exp(b*x)+c #exponential function
        #return a*x**2+b*x+c #quadratic function
        return a * x ** 3 + b * x ** 2 + c * x +  d  # quadratic function
        #return a*x+b #linear function

    def s(x,a,b):
        #return a*np.exp(b*x)+c #exponential function
        return a*x**2+b*x+c #quadratic function


    real_data = merged_df[merged_df["Date"] < "2023-11-08"]

    #fit the curve with x as the Day column and y as the cumulative sales
    #popt, pcov = curve_fit(f, merged_df["Day"], merged_df["Cumulative Sales"])

    # fit the curve with x as the Day column and y as the cumulative sales giving a good fist guess for an exponential with form a*np.exp(-b*x)+c
    popt, pcov = curve_fit(f, real_data["Day"], real_data["Cumulative Sales"],p0=[1,1,1,1])

    #find the day value associated with the last day of 2023
    lastDayOf2023 = merged_df[merged_df["Date"] == "2023-12-31"]["Day"].values[0]
    lastDayOf2024 = merged_df[merged_df["Date"] == "2024-12-31"]["Day"].values[0]

    opt, pcov = curve_fit(f, [lastDayOf2023,lastDayOf2023 + 1,lastDayOf2024-1,lastDayOf2024], [123,124,798,800], p0=[ 1.12605265e-06 ,-7.14422303e-04 , 1.51153830e-01, -2.78657347e+00])
    pess, pcov = curve_fit(f, [lastDayOf2023,lastDayOf2023 + 1,lastDayOf2024-1,lastDayOf2024], [120,120.5,499,500], p0=[ 1.12605265e-06, -7.14422303e-04 , 1.51153830e-01, -2.78657347e+00])

    
    predicted_data = merged_df[merged_df["Date"] > "2023-12-31"]

    predicted_data["Optimistic Curve"] = f(predicted_data["Day"],*opt)

    predicted_data["Pessimistic Curve"] = f(predicted_data["Day"],*pess)



    print(popt) # [ 1.12605265e-06 -7.14422303e-04  1.51153830e-01 -2.78657347e+00]
    #merged df less than 2023 11 08

    optimisticSacale = 2
    pessimisticScale = 0.5

    #make a new column that is the fitted curve
    merged_df["Normal Curve"] = f(merged_df["Day"],*popt)

    #optimistic params
    optimisticParams = popt.copy()
    #scale the last value in the array by the optimistic scale, ITS NOT A LIST
    optimisticParams[2] = optimisticParams[2] * optimisticSacale
    # make a new column that is the fitted curve
    merged_df["Optimistic Curve"] = f(merged_df["Day"], *optimisticParams)

    pessimisticParams = popt.copy()
    pessimisticParams[2] = pessimisticParams[2] * pessimisticScale

    # make a new column that is the fitted curve
    merged_df["Pessimistic Curve"] = f(merged_df["Day"], *pessimisticParams)

    #set y upper lim to max cumulative salaes val
    y_upper_lim = merged_df["Cumulative Sales"].max() * 10


    #store the value of the normal curve at the end of 2024
    endOf2024 = merged_df[merged_df["Date"] == "2024-12-31"]["Normal Curve"].values[0]

    # store the value of the normal curve at the start of 2023
    endOf2023 = merged_df[merged_df["Date"] == "2023-12-31"]["Normal Curve"].values[0]

    #store the value of the optimistic curve at the end of 2024
    endOf2024Optimistic = predicted_data[predicted_data["Date"] == "2024-12-31"]["Optimistic Curve"].values[0]

    #store the value of the pessimistic curve at the end of 2024
    endOf2024Pessimistic = predicted_data[predicted_data["Date"] == "2024-12-31"]["Pessimistic Curve"].values[0]


    #calculate



    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(real_data['Date'], real_data['Cumulative Sales'], color='blue', label='Coated Fleet',s = 6)
    plt.plot(merged_df['Date'], merged_df["Normal Curve"], color='blue', label='Best Fit Curve')
    plt.plot(predicted_data['Date'], predicted_data["Optimistic Curve"], color='green', label='Optimistic Curve')
    plt.plot(predicted_data['Date'], predicted_data["Pessimistic Curve"], color='red', label='Pessimistic Curve')

    #include text to the right of the plot that shows the value of the normal curve at the end of 2024 using endOf2024
    plt.text(x=pd.Timestamp(year=2025, month=1, day=15), y=endOf2024, s=str(round(endOf2024)) + "(" + str(round(endOf2024/endOf2023,2)) + "x 2023)", color='blue', verticalalignment='center', horizontalalignment='left', fontsize=15)

    #include text to the right of the plot that shows the value of the optimistic curve at the end of 2024 using endOf2024
    plt.text(x=pd.Timestamp(year=2025, month=1, day=15), y=endOf2024Optimistic, s=str(round(endOf2024Optimistic)) + "(" + str(round(endOf2024Optimistic/endOf2023,2)) + "x 2023)", color='green', verticalalignment='center', horizontalalignment='left', fontsize=15)

    #include text to the right of the plot that shows the value of the pessimistic curve at the end of 2024 using endOf2024
    plt.text(x=pd.Timestamp(year=2025, month=1, day=15), y=endOf2024Pessimistic, s=str(round(endOf2024Pessimistic)) + "(" + str(round(endOf2024Pessimistic/endOf2023,2)) + "x 2023)", color='red', verticalalignment='center', horizontalalignment='left', fontsize=15)

    #include text to the right of the plot that shows the value of the normal curve at the end of 2023 using endOf2023
    plt.text(x=pd.Timestamp(year=2023, month=12, day=31), y=endOf2023 -30, s=str(round(endOf2023)) + "\n(EOY 2023)", color='blue', verticalalignment='center', horizontalalignment='left', fontsize=15)


    #plot a vertival black line at the end of 2024

    plt.axvline(x=pd.Timestamp(year=2024, month=12, day=31), color='black',  label='End of 2024')

    #plot a vertival black line at the end of 2023
    plt.axvline(x=pd.Timestamp(year=2023, month=12, day=31), color='black',  linestyle='--', label='End of 2023')



    plt.ylim(0, y_upper_lim)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=20)
    #legend with large font size
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()


growth_data = pd.read_excel("growth.xlsx")

growth_plot(growth_data, 'Sale Date', title='GIT Coated Vessels\n2024 Prediction', xlabel='Date', ylabel='Fleet Size')

