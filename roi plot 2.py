import numpy as np
import matplotlib.pyplot as plt

# Debug mode
debug = False

import numpy as np
import pandas as pd

#start date is today, end date is 16 months from now
start_date = pd.to_datetime('today')
end_date = start_date + pd.DateOffset(months=16)

#prop_start is
prop_start = -15000

# Increase the resolution of the dataset to better capture the dip at 10 months
n_points_high_res = 1000
dates_high_res = pd.date_range(start_date, end_date, periods=n_points_high_res)
df_high_res = pd.DataFrame({'Timestamp': dates_high_res})
df_high_res['Months'] = (df_high_res['Timestamp'] - start_date).dt.days / 30

# Update the 'prop' data to have a linear dip at the 10-month mark
scaling_factor_high_res = abs(prop_start) / np.log1p(2)
prop_data_high_res = np.log1p(df_high_res['Months']) * scaling_factor_high_res + prop_start

# Find the point corresponding to 10 months
ten_month_point_high_res = np.where(df_high_res['Months'] >= 10)[0][0]

# Make the dip at 10 months linear (i.e., a vertical drop)
prop_data_high_res[ten_month_point_high_res:] = prop_data_high_res[ten_month_point_high_res-1] - 15000

# Reset the log behavior after the dip at 10 months
prop_data_high_res[ten_month_point_high_res:] = np.log1p(np.linspace(1, 16-10, n_points_high_res-ten_month_point_high_res)) * scaling_factor_high_res + prop_data_high_res[ten_month_point_high_res]

# Update the 'polishing' data
# The curve starts with a slow decrease and then increases the rate of decrease over time
# It restarts at the 6 and 12 month marks
polishing_data_high_res = -np.exp(df_high_res['Months']/6) * 1000

# Apply the dips at the 6 and 12 month marks
six_month_point_high_res = np.where(df_high_res['Months'] >= 6)[0][0]
twelve_month_point_high_res = np.where(df_high_res['Months'] >= 12)[0][0]
polishing_data_high_res[six_month_point_high_res:] += -10000
polishing_data_high_res[twelve_month_point_high_res:] += -10000

# Update the high-res DataFrame
df_high_res['prop'] = prop_data_high_res
df_high_res['polishing'] = polishing_data_high_res

# Save the updated high-res DataFrame to CSV
csv_path_high_res = 'prop_polishing_data_high_res.csv'
df_high_res.to_csv(csv_path_high_res, index=False)

# Plot the updated high-res data for debugging
plt.figure(figsize=(14, 7))
plt.plot(df_high_res['Months'], df_high_res['prop'], label='XGIT-PROP',linewidth = 5)
plt.plot(df_high_res['Months'], df_high_res['polishing'] + 700, label='Polishing', color='r',linewidth = 5)

# Proactive cleaning zone
plt.axvspan(8, 12, alpha=0.2, color='grey')

arrowFontSize = 15

# Corrected annotation for the plot at the 10-month mark
plt.annotate('Soft Grooming\n      $7.5-10k', xy=(10, df_high_res['prop'].iloc[ten_month_point_high_res]),
             xytext=(10, df_high_res['prop'].iloc[ten_month_point_high_res] -10000), arrowprops=dict(facecolor='black', shrink=0.05),fontsize = arrowFontSize)


# Corrected annotation for the plot at the 10-month mark
plt.annotate('Prop Polishing\n     $-10,000', xy=(6, df_high_res['polishing'].iloc[ten_month_point_high_res]+3000),
             xytext=(3, df_high_res['polishing'].iloc[ten_month_point_high_res] - 4000), arrowprops=dict(facecolor='black', shrink=0.05),fontsize = arrowFontSize)



# Corrected annotation for the plot at the 10-month mark
plt.annotate('Prop Polishing\n     $-10,000', xy=(12, df_high_res['polishing'].iloc[ten_month_point_high_res]-12000),
             xytext=(13, df_high_res['prop'].iloc[ten_month_point_high_res] -23000), arrowprops=dict(facecolor='black', shrink=0.05),fontsize = arrowFontSize)


# Adding text at the bottom in the middle of the grey zone
mid_point_grey_zone = (8 + 12) / 2
plt.text(mid_point_grey_zone, min(df_high_res['prop'].min() - 5000, df_high_res['polishing'].min()), 'Proactive Cleaning Zone', horizontalalignment='center', verticalalignment='bottom', fontsize=15, style='italic')


# Annotations and titles
plt.axhline(y=0, color='black',linewidth=0.5,linestyle = "--")
plt.title("XGIT PROP vs Polishing",fontsize = 30)
plt.xlabel("Time (months)",fontsize = 20)
plt.ylabel("ROI ($)", fontsize=20, rotation=0)
plt.legend(fontsize = 20)
plt.yticks([-15000,0], fontsize=16)  # Keep only the -15000 y-tick

# Add italicized text underneath the -15,000 tick
plt.text(-0.9, -17000, 'Average XGIT-PROP Cost', horizontalalignment='right', verticalalignment='top', fontsize=12, style='italic')  # <--- Add this line here


disclamerSize = 9
# Add italicized text underneath the -15,000 tick
plt.text(-4.1, 30000, '*XGIT-PROP offers better efficiency and ROI in\n2-3 months compared to traditional prop \npolishing every 6 months.', horizontalalignment='left', verticalalignment='top', fontsize=disclamerSize, style='italic')  # <--- Add this line here
# Add italicized text underneath the -15,000 tick
plt.text(-4.1, 22000, '*Maintenance is easier with XGIT-PROP,\nneeding only soft grooming every 8-12 months\nfor optimal performance.', horizontalalignment='left', verticalalignment='top', fontsize=disclamerSize, style='italic')  # <--- Add this line here
# Add italicized text underneath the -15,000 tick
plt.text(-4.1, 14000, '*A 30,000 DWT Oil Tanker can save around\n$28K USD in fuel over\n12 months with XGIT-PROP.', horizontalalignment='left', verticalalignment='top', fontsize=disclamerSize, style='italic')  # <--- Add this line here

plt.text(-4.1, 8000, '*Over a 30-month period, a shipowner can\n save $80K USD from\nfuel and reduced cleanings', horizontalalignment='left', verticalalignment='top', fontsize=disclamerSize, style='italic')  # <--- Add this line here


plt.xticks(fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Display the plot
plt.show()

