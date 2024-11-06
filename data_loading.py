import streamlit as st
import pandas as pd
import numpy as np

st.markdown('''**PDATA**''')
st.markdown('''''')
st.markdown('''''')
st.markdown('''---''')
st.markdown('''''')
st.markdown('''''')
st.markdown('''Purchasing Data 2024 of all regions Uploaded into Commande BO shared drive''')

# @title
# Import necessary libraries
import pandas as pd
from pandas.tseries.offsets import DateOffset

# Mount Google Drive (already mounted in your case)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# File paths
orders_sept_path = '/content/drive/Shareddrives/Commandes BO/2024.xlsx'
pdata_sept_path = '/content/drive/Shareddrives/Commandes BO/PDATA.xlsx'
segmentation_path = '/content/drive/Shareddrives/Commandes BO/segmentation.xlsx'

# Load the datasets
orders_sept = pd.read_excel(orders_sept_path).dropna(how='all')
pdata_sept = pd.read_excel(pdata_sept_path).dropna(how='all')
segmentation_df = pd.read_excel(segmentation_path).dropna(how='all')

# Ensure that the date columns are properly converted to datetime
if 'Date de commande' in orders_sept.columns:
    orders_sept['Date de commande'] = pd.to_datetime(orders_sept['Date de commande'], errors='coerce')

if 'Date' in pdata_sept.columns:
    pdata_sept['Date'] = pd.to_datetime(pdata_sept['Date'], errors='coerce')

# Calculate the current month and the last two months using Timestamp and then converting to Period
current_month = pd.Timestamp.now().to_period('M')
last_month = (pd.Timestamp.now() - DateOffset(months=1)).to_period('M')
two_months_ago = (pd.Timestamp.now() - DateOffset(months=2)).to_period('M')

# Store the last three months
months = [current_month, last_month, two_months_ago]

# Initialize an empty list to store processed data for each month
monthly_results = []

# Process each month
for month in months:
    try:
        # Step 1: Filter orders and pdata by month (now that date columns are datetime)
        if 'Date de commande' in orders_sept.columns and 'Date' in pdata_sept.columns:
            orders_filtered = orders_sept[orders_sept['Date de commande'].dt.to_period('M') == month]
            pdata_filtered = pdata_sept[pdata_sept['Date'].dt.to_period('M') == month]
        else:
            print(f"'Date de commande' or 'Date' column not found in the dataset for {month}. Skipping this month.")
            continue

        # Check if the unique 'Reference' matches the unique 'order_id' in pdata
        unique_orders = orders_filtered['Reference'].nunique()
        unique_order_ids = pdata_filtered['order_id'].nunique()

        if unique_orders == unique_order_ids:
            print(f"Unique References and Order IDs match for {month}: {unique_orders}")
        else:
            print(f"Mismatch between unique References ({unique_orders}) and Order IDs ({unique_order_ids}) for {month}")

        # Step 2: Pivot 'Product Category' column in pdata to separate binary columns (one-hot encoding)
        pdata_filtered.loc[:, 'Product Category'] = pdata_filtered['Product Category'].fillna('Unknown').astype(str)
        pdata_pivoted = pd.get_dummies(pdata_filtered[['order_id', 'Product Category']], columns=['Product Category'])

        # Ensure 'order_id' is the first column and product categories follow it
        pdata_pivoted = pdata_pivoted.groupby('order_id').max().reset_index()

        # Sort product category columns to be next to each other
        product_category_columns = [col for col in pdata_pivoted.columns if 'Product Category' in col]
        pdata_pivoted = pdata_pivoted[['order_id'] + sorted(product_category_columns)]

        # Step 3: Merge filtered orders with the pivoted product categories on 'Reference' and 'order_id'
        merged_df = pd.merge(orders_filtered, pdata_pivoted, left_on='Reference', right_on='order_id', how='left')

        # Step 4: Calculate GMV euro based on 'Total' and 'Pays' (country)
        def calculate_gmv(row):
            if row['Pays'] == 'US':
                return row['Total'] * 0.95
            elif row['Pays'] == 'GB':
                return row['Total'] * 1.15
            else:
                return row['Total']

        merged_df['GMV euro'] = merged_df.apply(calculate_gmv, axis=1)

        # Step 5: Merge the segmentation data based on 'Restaurant ID'
        final_with_segmentation = pd.merge(
            merged_df,
            segmentation_df[['Restaurant_id', 'Gamme', 'Catégorie', 'Type']],
            left_on='Restaurant ID',
            right_on='Restaurant_id',
            how='left'
        )

        # Step 6: Add the 'region' column from pdata based on 'Reference' and 'order_id'
        final_with_region = pd.merge(
            final_with_segmentation,  # This is the dataframe with segmentation already merged
            pdata_filtered[['order_id', 'region']],  # We take 'order_id' and 'region' columns from Pdata
            left_on='Reference',  # We match the 'Reference' column from Orders
            right_on='order_id',  # We match it with 'order_id' from Pdata
            how='left'  # Left join to keep all orders
        )

        # Step 7: Add the 'Month' column for reference
        final_with_region['Month'] = str(month)

        # Step 8: Ensure all product category columns are binary (1 or 0)
        product_columns = [col for col in final_with_region.columns if 'Product Category' in col]
        final_with_region[product_columns] = final_with_region[product_columns].fillna(0).astype(int)

        # Step 9: Remove rows where all product category columns have 0 (i.e., rows with no product categories)
        final_with_region = final_with_region[final_with_region[product_columns].sum(axis=1) > 0]

        # Check if there are any NaN values
        if final_with_region.isna().sum().sum() == 0:
            print(f"No NaN values in the dataset for {month}")
        else:
            print(f"There are NaN values in the dataset for {month}. Please investigate.")

        # Step 10: Group by 'Reference' to avoid duplicates and take the first occurrence of each column
        grouped_final = final_with_region.groupby('Reference').first().reset_index()

        # Step 11: Ensure the product category columns are placed right after 'Month'
        # Move product category columns to be after 'Month'
        columns_order = list(grouped_final.columns)
        product_category_start_idx = columns_order.index('Month') + 1  # Product category columns should follow 'Month'

        # Remove product category columns from their current location and insert them right after 'Month'
        for col in product_columns:
            columns_order.remove(col)

        # Insert product category columns back after 'Month'
        columns_order[product_category_start_idx:product_category_start_idx] = product_columns

        # Reorder the dataframe columns
        grouped_final = grouped_final[columns_order]

        # Check if data was processed, then append to results
        if not grouped_final.empty:
            print(f"Processed data for {month}: {len(grouped_final)} rows")
            monthly_results.append(grouped_final)

            # Save each month's results into a separate file
            output_path = f'/content/drive/Shareddrives/Commandes BO/final_with_region_and_segmentation_{month}.xlsx'
            grouped_final.to_excel(output_path, index=False)
            print(f"File saved for {month}: {output_path}")
        else:
            print(f"No data for {month}")

    except Exception as e:
        print(f"Failed to process data for {month}: {e}")

# After loop: If you need to concatenate all results, you can do so like this:
if monthly_results:
    # Filter non-empty dataframes and concatenate
    non_empty_results = [df for df in monthly_results if not df.empty]

    if non_empty_results:
        final_combined = pd.concat(non_empty_results, ignore_index=True)
        combined_output_path = '/content/drive/Shareddrives/Commandes BO/PDATA_Orders.xlsx'
        final_combined.to_excel(combined_output_path, index=False)
        print(f"Final combined file saved to: {combined_output_path}")
    else:
        print("No non-empty dataframes to concatenate.")
else:
    print("No monthly data was processed.")


# @title
import pandas as pd

# Load the dataset (update the path to where your file is located)
file_path = '/content/drive/Shareddrives/Commandes BO/PDATA_Orders.xlsx'

# Load the dataset
df = pd.read_excel(file_path)

# List of product categories with correct column names
product_categories = ['Product Category_Boissons', 'Product Category_Boucherie', 'Product Category_Charcuterie',
                      'Product Category_Consommables', 'Product Category_Crémerie', 'Product Category_Emballages',
                      'Product Category_Epicerie salée', 'Product Category_Epicerie sucrée',
                      'Product Category_Fruits et Légumes', 'Product Category_Marée',
                      'Product Category_Produits Préparés']

# Step 1: Create an 'Area Code' column based on the 'Pays' and 'Postal code' columns
def extract_area_code(postal_code, pays):
    if pays == 'GB':
        for i, char in enumerate(postal_code):
            if char.isdigit():
                return postal_code[:i]
        return postal_code  # fallback if no number is found
    else:
        return postal_code

# Check if the 'Area Code' column already exists and remove it if it does
if 'Area Code' in df.columns:
    df = df.drop(columns=['Area Code'])

# Apply the function row-wise to create 'Area Code' column and insert it right after 'Postal code'
df.insert(df.columns.get_loc('Postal code') + 1, 'Area Code', df.apply(lambda row: extract_area_code(row['Postal code'], row['Pays']), axis=1))

# Step 2: Group the data by unique Restaurant ID
grouped_df_by_restaurant = df.drop_duplicates(subset='Restaurant ID')

# Step 3: Calculate the number of clients that purchased each category by Area Code and Pays
def calculate_clients_purchased(df, category):
    return df.groupby(['Area Code', 'Pays'])[category].transform(lambda x: (x == 1).sum())

for category in product_categories:
    grouped_df_by_restaurant[f'{category}_clients'] = calculate_clients_purchased(grouped_df_by_restaurant, category)

# Step 4: Calculate the least average basket for each product category by Area Code and Pays
def calculate_least_avg_basket(df, category):
    grouped = df[df[category] == 1].groupby(['Area Code', 'Pays'])['GMV euro'].mean()
    return df.set_index(['Area Code', 'Pays']).index.map(grouped)

for category in product_categories:
    grouped_df_by_restaurant[f'{category}_least_avg_basket'] = calculate_least_avg_basket(grouped_df_by_restaurant, category)

# Step 5: Calculate the number of clients that purchased each category by Area Code, Pays, region, and 'Catégorie'
def calculate_clients_purchased_by_categorie(df, category):
    return df.groupby(['Area Code', 'Pays', 'region', 'Catégorie'])[category].transform(lambda x: (x == 1).sum())

# Step 6: Calculate the least average basket for each product category by Area Code, Pays, region, and 'Catégorie'
def calculate_least_avg_basket_by_categorie(df, category):
    grouped = df[df[category] == 1].groupby(['Area Code', 'Pays', 'region', 'Catégorie'])['GMV euro'].mean()
    return df.set_index(['Area Code', 'Pays', 'region', 'Catégorie']).index.map(grouped)

# Apply the functions for each product category
for category in product_categories:
    grouped_df_by_restaurant[f'{category}_clients_categorie'] = calculate_clients_purchased_by_categorie(grouped_df_by_restaurant, category)
    grouped_df_by_restaurant[f'{category}_least_avg_basket_categorie'] = calculate_least_avg_basket_by_categorie(grouped_df_by_restaurant, category)

# Step 7: Add a new column that contains the count of product categories purchased
grouped_df_by_restaurant['Product_Categories_Purchased_Count'] = grouped_df_by_restaurant[product_categories].sum(axis=1)

# Step 8: Save the final result, ensuring that data is grouped by unique Restaurant ID
output_path = '/content/drive/Shareddrives/Commandes BO/Clients_Purhcases.xlsx'
grouped_df_by_restaurant.to_excel(output_path, index=False)

print(f"File saved to: {output_path}")

# @title
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Step 2: Load the dataset from Google Drive
file_path = '/content/drive/Shareddrives/Commandes BO/2024.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Step 2: Data Preparation
df['Date de commande'] = pd.to_datetime(df['Date de commande'], errors='coerce')
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

# Step 3: Calculate GMV euro based on 'Total' and 'Pays' (country)
def calculate_gmv(row):
    if row['Pays'] == 'US':
        return row['Total'] * 0.95
    elif row['Pays'] == 'GB':
        return row['Total'] * 1.15
    else:
        return row['Total']

df['GMV euro'] = df.apply(calculate_gmv, axis=1)

# Define filtered_df (excluding first-time purchases)
filtered_df = df.copy()

# Identify first-time purchases and filter them out
first_purchase_dates = df.groupby('Restaurant ID')['Date de commande'].min().reset_index()
first_purchase_dates.columns = ['Restaurant ID', 'First Purchase Date']
filtered_df = filtered_df.merge(first_purchase_dates, on='Restaurant ID')
filtered_df = filtered_df[filtered_df['Date de commande'] != filtered_df['First Purchase Date']]

# Part 1: Monthly Average Spend and Active Customers with Sorted Month Names
filtered_df['Month'] = filtered_df['Date de commande'].dt.to_period('M')

# Convert period to month names for better visualization
filtered_df['Month_Name'] = filtered_df['Date de commande'].dt.strftime('%B')
filtered_df['Month_Year'] = filtered_df['Date de commande'].dt.to_period('M')

# Sort the data by month based on month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
filtered_df['Month_Name'] = pd.Categorical(filtered_df['Month_Name'], categories=month_order, ordered=True)
filtered_df = filtered_df.sort_values('Month_Year')
existing_months = filtered_df['Month_Year'].unique()

# Calculate total spend and active customers per month
total_spend_per_month = filtered_df.groupby('Month_Year')['GMV euro'].sum()
unique_clients_per_month = filtered_df.groupby('Month_Year')['Restaurant ID'].nunique()
average_spend_per_client_per_month = total_spend_per_month / unique_clients_per_month

# Create the interactive bar chart for average spend and active customers
fig1 = go.Figure()

# Add the average spend per client (bars)
fig1.add_trace(go.Bar(
    x=average_spend_per_client_per_month.index.strftime('%B %Y'),  # Use month-year names for x-axis
    y=average_spend_per_client_per_month,
    name='Average Spend per Client (€)',
    text=[f'{val:.2f}€' for val in average_spend_per_client_per_month],
    textposition='auto',
    marker_color='#0F8B8D'
))

# Add the active customers (line)
fig1.add_trace(go.Scatter(
    x=unique_clients_per_month.index.strftime('%B %Y'),  # Use month-year names for x-axis
    y=unique_clients_per_month,
    name='Number of Active Customers',
    mode='lines+markers',
    marker=dict(color='#EC9A29', size=10),
    line=dict(color='#EC9A29', width=2)
))

# Update layout
fig1.update_layout(
    title='Average Spend Per Client Per Month (Excluding First Time Purchases)',
    xaxis_title='Month',
    yaxis_title='Average Spend (€)',
    yaxis2=dict(title='Number of Active Customers', overlaying='y', side='right'),
    barmode='group',
    template='plotly_white'
)

# Show the figure
fig1.show()

# Part 1b: Monthly Average Spend and Active Customers Including First Time Purchases
# Add Month_Name column to df
df['Month_Name'] = df['Date de commande'].dt.strftime('%B')
df['Month_Year'] = df['Date de commande'].dt.to_period('M')

# Sort the data by month based on month order
df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
df = df.sort_values('Month_Year')
existing_months_incl = df['Month_Year'].unique()

# Calculate total spend and active customers per month including first time purchases
total_spend_per_month_incl = df.groupby('Month_Year')['GMV euro'].sum()
unique_clients_per_month_incl = df.groupby('Month_Year')['Restaurant ID'].nunique()
average_spend_per_client_per_month_incl = total_spend_per_month_incl / unique_clients_per_month_incl

# Create the interactive bar chart for average spend and active customers including first time purchases
fig1b = go.Figure()

# Add the average spend per client (bars)
fig1b.add_trace(go.Bar(
    x=average_spend_per_client_per_month_incl.index.strftime('%B %Y'),  # Use month-year names for x-axis
    y=average_spend_per_client_per_month_incl,
    name='Average Spend per Client (€) (Including First Time Purchases)',
    text=[f'{val:.2f}€' for val in average_spend_per_client_per_month_incl],
    textposition='auto',
    marker_color='#0F8B8D'
))

# Add the active customers (line)
fig1b.add_trace(go.Scatter(
    x=unique_clients_per_month_incl.index.strftime('%B %Y'),  # Use month-year names for x-axis
    y=unique_clients_per_month_incl,
    name='Number of Active Customers (Including First Time Purchases)',
    mode='lines+markers',
    marker=dict(color='#EC9A29', size=10),
    line=dict(color='#EC9A29', width=2)
))

# Update layout
fig1b.update_layout(
    title='Average Spend Per Client Per Month (Including First Time Purchases)',
    xaxis_title='Month',
    yaxis_title='Average Spend (€)',
    yaxis2=dict(title='Number of Active Customers', overlaying='y', side='right'),
    barmode='group',
    template='plotly_white'
)

# Show the figure
fig1b.show()

# Part 2: Monthly Percentage Change in Average Spend (Same Time Frame as October MTD)
days_in_october_mtd = 7  # Use actual day count for October MTD
months_to_process = filtered_df['Month_Year'].unique()
average_spend_per_customer_same_days = {}

for month in months_to_process:
    month_data = filtered_df[(filtered_df['Month_Year'] == month) &
                             (filtered_df['Date de commande'].dt.day <= days_in_october_mtd)]
    total_spend = month_data['GMV euro'].sum()
    active_customers = month_data['Restaurant ID'].nunique()
    average_spend_per_customer_same_days[month.strftime('%Y-%m')] = total_spend / active_customers

average_spend_per_customer_same_days_series = pd.Series(average_spend_per_customer_same_days)

# Calculate percentage change
percentage_change = average_spend_per_customer_same_days_series.pct_change() * 100

# Create the interactive line chart for monthly percentage change
fig2 = go.Figure()

# Add the average spend per customer (line)
fig2.add_trace(go.Scatter(
    x=average_spend_per_customer_same_days_series.index,
    y=average_spend_per_customer_same_days_series,
    name='Average Spend per Customer (€)',
    mode='lines+markers',
    marker=dict(color='#0F8B8D', size=10),
    line=dict(color='#0F8B8D', width=2)
))

# Add percentage change as text labels (green for positive, red for negative)
for i, month in enumerate(percentage_change.index):
    if i > 0:  # Skip the first element
        change = percentage_change.iloc[i]
        color = 'green' if change > 0 else 'red'
        fig2.add_annotation(
            x=month,
            y=average_spend_per_customer_same_days_series.iloc[i],
            text=f'{change:.2f}%',
            showarrow=True,
            arrowhead=2,
            font=dict(color=color)
        )

# Update layout
fig2.update_layout(
    title='Average Spend Per Customer (Same Time Frame as October MTD)',
    xaxis_title='Month',
    yaxis_title='Average Spend (€)',
    template='plotly_white'
)

# Show the figure
fig2.show()

# Part 3: Weekly Average Spend per Client and Percentage Change with ISO Week Numbers
filtered_df['Week'] = filtered_df['Date de commande'].dt.isocalendar().week
filtered_df['Year'] = filtered_df['Date de commande'].dt.isocalendar().year
filtered_df['Week_Order'] = filtered_df['Year'].astype(str) + '-W' + filtered_df['Week'].astype(str)
existing_weeks = filtered_df['Week_Order'].unique()

# Calculate total spend and unique clients per week
total_spend_per_week = filtered_df.groupby('Week_Order')['GMV euro'].sum()
unique_clients_per_week = filtered_df.groupby('Week_Order')['Restaurant ID'].nunique()
average_spend_per_client_per_week = total_spend_per_week / unique_clients_per_week

# Create the interactive line chart for weekly average spend
fig3 = go.Figure()

# Add the average spend per client (line)
fig3.add_trace(go.Scatter(
    x=average_spend_per_client_per_week.index,  # Use existing weeks
    y=average_spend_per_client_per_week,
    name='Weekly Average Spend per Client (€)',
    mode='lines+markers',
    marker=dict(color='#0F8B8D', size=10),
    line=dict(color='#0F8B8D', width=2)
))

# Update layout
fig3.update_layout(
    title='Weekly Average Spend Per Client (Linear)',
    xaxis_title='ISO Week Number',
    yaxis_title='Average Spend (€)',
    template='plotly_white'
)

# Show the figure
fig3.show()

# Calculate weekly percentage change
percentage_change_weekly = average_spend_per_client_per_week.pct_change() * 100

# Create the interactive line chart for weekly percentage change
fig4 = go.Figure()

# Add the percentage change (line)
fig4.add_trace(go.Scatter(
    x=percentage_change_weekly.index,  # Use existing weeks
    y=percentage_change_weekly,
    name='Weekly Percentage Change in Average Spend',
    mode='lines+markers',
    marker=dict(color='#0F8B8D', size=10),
    line=dict(color='#0F8B8D', width=2)
))

# Add percentage change labels (green for positive, red for negative)
for i, week in enumerate(percentage_change_weekly.index):
    if i > 0:  # Skip the first element
        change = percentage_change_weekly.iloc[i]
        color = 'green' if change > 0 else 'red'
        fig4.add_annotation(
            x=week,
            y=percentage_change_weekly.iloc[i],
            text=f'{change:.2f}%',
            showarrow=True,
            arrowhead=2,
            font=dict(color=color)
        )

# Update layout
fig4.update_layout(
    title='Weekly Percentage Change in Average Spend Per Client',
    xaxis_title='ISO Week Number',
    yaxis_title='Percentage Change (%)',
    template='plotly_white'
)

# Show the figure
fig4.show()

# Part 4: Waterfall Chart for MTD Changes in Average Spend Components (Frequency, Average Basket)

# Step 1: Define the data for Last Month and This Month MTD
last_month_name = filtered_df['Month_Year'].unique()[-2]  # Dynamically get the name of last month
current_month_name = filtered_df['Month_Year'].unique()[-1]  # Dynamically get the name of current month

data = {
    'Last Month': {
        'Frequency': filtered_df[filtered_df['Month_Year'] == filtered_df['Month_Year'].unique()[-2]].shape[0] / unique_clients_per_month.iloc[-2],
        'Average Basket': total_spend_per_month.iloc[-2] / filtered_df[filtered_df['Month_Year'] == filtered_df['Month_Year'].unique()[-2]].shape[0]
    },
    'This Month MTD': {
        'Frequency': filtered_df[filtered_df['Month_Year'] == filtered_df['Month_Year'].unique()[-1]].shape[0] / unique_clients_per_month.iloc[-1],
        'Average Basket': total_spend_per_month.iloc[-1] / filtered_df[filtered_df['Month_Year'] == filtered_df['Month_Year'].unique()[-1]].shape[0]
    }
}

# Step 2: Calculate last month's average spend (Orders * Frequency)
last_month_avg_spend = average_spend_per_client_per_month.iloc[-2]
current_month_avg_spend = average_spend_per_client_per_month.iloc[-1]

# Step 3: Calculate percentage changes
frequency_change_pct = (data['This Month MTD']['Frequency'] - data['Last Month']['Frequency']) / data['Last Month']['Frequency']
average_basket_change_pct = (data['This Month MTD']['Average Basket'] - data['Last Month']['Average Basket']) / data['Last Month']['Average Basket']

# Step 4: Calculate the contributions of each factor to the change in average spend
frequency_contribution = frequency_change_pct * last_month_avg_spend
average_basket_contribution = average_basket_change_pct * last_month_avg_spend

# Step 5: Total change in average spend (calculated from the individual contributions)
total_change = frequency_contribution + average_basket_contribution

# Step 6: Create the Waterfall Chart
fig5 = go.Figure(go.Waterfall(
    name="Average Spend Change",
    orientation="v",
    measure=["absolute", "relative", "relative", "absolute"],
    x=[f"{last_month_name} Avg Spend", "Frequency Contribution", "Average Basket Contribution", f"{current_month_name} Avg Spend"],
    textposition="outside",
    text=[f"{last_month_avg_spend:.2f}€", f"{frequency_contribution:.2f}€", f"{average_basket_contribution:.2f}€", f"{current_month_avg_spend:.2f}€"],
    y=[last_month_avg_spend, frequency_contribution, average_basket_contribution, current_month_avg_spend],
    connector={"line": {"color": "rgb(63, 63, 63)"}}
))

# Add chart title and labels
fig5.update_layout(
    title=f"Waterfall Chart of Average Spend Change ({last_month_name} to {current_month_name})",
    xaxis_title="Components",
    yaxis_title="Contribution (€)",
    waterfallgap=0.3
)

# Show the chart
fig5.show()

# Part 5: Waterfall Chart for MTD Changes Including First Purchases
# Step 1: Define the data for Last Month and This Month MTD Including First Purchases
last_month_name_incl = df['Month_Year'].unique()[-2]  # Dynamically get the name of last month
current_month_name_incl = df['Month_Year'].unique()[-1]  # Dynamically get the name of current month

data_incl = {
    'Last Month': {
        'Frequency': df[df['Month_Year'] == df['Month_Year'].unique()[-2]].shape[0] / unique_clients_per_month_incl.iloc[-2],
        'Average Basket': total_spend_per_month_incl.iloc[-2] / df[df['Month_Year'] == df['Month_Year'].unique()[-2]].shape[0]
    },
    'This Month MTD': {
        'Frequency': df[df['Month_Year'] == df['Month_Year'].unique()[-1]].shape[0] / unique_clients_per_month_incl.iloc[-1],
        'Average Basket': total_spend_per_month_incl.iloc[-1] / df[df['Month_Year'] == df['Month_Year'].unique()[-1]].shape[0]
    }
}

# Step 2: Calculate last month's average spend (Orders * Frequency)
last_month_avg_spend_incl = average_spend_per_client_per_month_incl.iloc[-2]
current_month_avg_spend_incl = average_spend_per_client_per_month_incl.iloc[-1]

# Step 3: Calculate percentage changes
frequency_change_pct_incl = (data_incl['This Month MTD']['Frequency'] - data_incl['Last Month']['Frequency']) / data_incl['Last Month']['Frequency']
average_basket_change_pct_incl = (data_incl['This Month MTD']['Average Basket'] - data_incl['Last Month']['Average Basket']) / data_incl['Last Month']['Average Basket']

# Step 4: Calculate the contributions of each factor to the change in average spend
frequency_contribution_incl = frequency_change_pct_incl * last_month_avg_spend_incl
average_basket_contribution_incl = average_basket_change_pct_incl * last_month_avg_spend_incl

# Step 5: Total change in average spend (calculated from the individual contributions)
total_change_incl = frequency_contribution_incl + average_basket_contribution_incl

# Step 6: Create the Waterfall Chart
fig6 = go.Figure(go.Waterfall(
    name="Average Spend Change (Including First Purchases)",
    orientation="v",
    measure=["absolute", "relative", "relative", "absolute"],
    x=[f"{last_month_name_incl} Avg Spend", "Frequency Contribution", "Average Basket Contribution", f"{current_month_name_incl} Avg Spend"],
    textposition="outside",
    text=[f"{last_month_avg_spend_incl:.2f}€", f"{frequency_contribution_incl:.2f}€", f"{average_basket_contribution_incl:.2f}€", f"{current_month_avg_spend_incl:.2f}€"],
    y=[last_month_avg_spend_incl, frequency_contribution_incl, average_basket_contribution_incl, current_month_avg_spend_incl],
    connector={"line": {"color": "rgb(63, 63, 63)"}}
))

# Add chart title and labels
fig6.update_layout(
    title=f"Waterfall Chart of Average Spend Change Including First Purchases ({last_month_name_incl} to {current_month_name_incl})",
    xaxis_title="Components",
    yaxis_title="Contribution (€)",
    waterfallgap=0.3
)

# Show the chart
fig6.show()


# @title
import pandas as pd
import plotly.graph_objects as go
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Step 2: Load the dataset from Google Drive
file_path = '/content/drive/Shareddrives/Commandes BO/PDATA_Orders.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Step 2: Data Preparation
df['Date de commande'] = pd.to_datetime(df['Date de commande'], errors='coerce')
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

# Step 3: Calculate GMV euro based on 'Total' and 'Pays' (country)
def calculate_gmv(row):
    if row['Pays'] == 'US':
        return row['Total'] * 0.95
    elif row['Pays'] == 'GB':
        return row['Total'] * 1.15
    else:
        return row['Total']

df['GMV euro'] = df.apply(calculate_gmv, axis=1)

# Define filtered_df (excluding first-time purchases)
filtered_df = df.copy()

# Identify first-time purchases and filter them out
first_purchase_dates = df.groupby('Restaurant ID')['Date de commande'].min().reset_index()
first_purchase_dates.columns = ['Restaurant ID', 'First Purchase Date']
filtered_df = filtered_df.merge(first_purchase_dates, on='Restaurant ID')
filtered_df = filtered_df[filtered_df['Date de commande'] != filtered_df['First Purchase Date']]

# Filter by region
groups_by_region = df.groupby('region')

# Overall Graph: This Month Average Spend per Region and Percentage Change Compared to Last Month
current_month = df['Date de commande'].dt.to_period('M').max()
last_month = current_month - 1

total_spend_current_month = df[df['Date de commande'].dt.to_period('M') == current_month].groupby('region')['GMV euro'].sum()
total_spend_last_month = df[df['Date de commande'].dt.to_period('M') == last_month].groupby('region')['GMV euro'].sum()

unique_clients_current_month = df[df['Date de commande'].dt.to_period('M') == current_month].groupby('region')['Restaurant ID'].nunique()
unique_clients_last_month = df[df['Date de commande'].dt.to_period('M') == last_month].groupby('region')['Restaurant ID'].nunique()

average_spend_current_month = total_spend_current_month / unique_clients_current_month
average_spend_last_month = total_spend_last_month / unique_clients_last_month

percentage_change_mtd = ((average_spend_current_month - average_spend_last_month) / average_spend_last_month) * 100

# Create the combined bar and line chart
fig_overall = go.Figure()

# Add average spend per region (bars)
fig_overall.add_trace(go.Bar(
    x=average_spend_current_month.index,
    y=average_spend_current_month,
    name='Average Spend per Region (€) - Current Month',
    marker_color='#0F8B8D',
    text=[f'{val:.2f}€' for val in average_spend_current_month],
    textposition='auto'
))

# Add percentage change MTD (line)
fig_overall.add_trace(go.Scatter(
    x=percentage_change_mtd.index,
    y=percentage_change_mtd,
    name='Percentage Change MTD (%)',
    mode='lines+markers+text',
    text=[f'{val:.2f}%' for val in percentage_change_mtd],
    textposition='top center',
    yaxis='y2',
    marker=dict(color='#EC9A29', size=10),
    line=dict(color='#EC9A29', width=2),
    textfont=dict(color='#EC9A29')
))

# Update layout
fig_overall.update_layout(
    title='This Month Average Spend per Region and Percentage Change vs M-1',
    xaxis_title='Region',
    yaxis_title='Average Spend (€)',
    yaxis2=dict(title='Percentage Change (%)', overlaying='y', side='right'),
    barmode='group',
    template='plotly_white'
)

# Show the overall figure
fig_overall.show()

for region, region_df in groups_by_region:
    # Part 1: Monthly Average Spend and Active Customers with Sorted Month Names
    region_df['Month'] = region_df['Date de commande'].dt.to_period('M')

    # Convert period to month names for better visualization
    region_df['Month_Name'] = region_df['Date de commande'].dt.strftime('%B')
    region_df['Month_Year'] = region_df['Date de commande'].dt.to_period('M')

    # Sort the data by month based on month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    region_df['Month_Name'] = pd.Categorical(region_df['Month_Name'], categories=month_order, ordered=True)
    region_df = region_df.sort_values('Month_Year')
    existing_months = region_df['Month_Year'].unique()

    # Calculate total spend and active customers per month
    total_spend_per_month = region_df.groupby('Month_Year')['GMV euro'].sum()
    unique_clients_per_month = region_df.groupby('Month_Year')['Restaurant ID'].nunique()
    average_spend_per_client_per_month = total_spend_per_month / unique_clients_per_month

    # Create the interactive bar chart for average spend and active customers
    fig1 = go.Figure()

    # Add the average spend per client (bars)
    fig1.add_trace(go.Bar(
        x=average_spend_per_client_per_month.index.strftime('%B %Y'),  # Use month-year names for x-axis
        y=average_spend_per_client_per_month,
        name='Average Spend per Client (€)',
        text=[f'{val:.2f}€' for val in average_spend_per_client_per_month],
        textposition='auto',
        marker_color='#0F8B8D'
    ))

    # Add the active customers (line)
    fig1.add_trace(go.Scatter(
        x=unique_clients_per_month.index.strftime('%B %Y'),  # Use month-year names for x-axis
        y=unique_clients_per_month,
        name='Number of Active Customers',
        mode='lines+markers+text',
        text=[f'{val}' for val in unique_clients_per_month],
        textposition='top center',
        marker=dict(color='#EC9A29', size=10),
        line=dict(color='#EC9A29', width=2),
        textfont=dict(color='#EC9A29')
    ))

    # Update layout
    fig1.update_layout(
        title=f'Average Spend Per Client Per Month for {region}',
        xaxis_title='Month',
        yaxis_title='Average Spend (€)',
        yaxis2=dict(title='Number of Active Customers', overlaying='y', side='right'),
        barmode='group',
        template='plotly_white'
    )

    # Show the figure
    fig1.show()

    # Part 2: Monthly Percentage Change in Average Spend (Same Time Frame as October MTD)
    days_in_current_month_mtd = df['Date de commande'].max().day  # Use actual day count for October MTD
    months_to_process = region_df['Month_Year'].unique()
    average_spend_per_customer_same_days = {}

    for month in months_to_process:
        month_data = region_df[(region_df['Month_Year'] == month) &
                               (region_df['Date de commande'].dt.day <= days_in_current_month_mtd)]
        total_spend = month_data['GMV euro'].sum()
        active_customers = month_data['Restaurant ID'].nunique()
        if month in average_spend_last_month.index:
            average_spend_per_customer_same_days[month.strftime('%Y-%m')] = average_spend_last_month.loc[month]
        else:
            average_spend_per_customer_same_days[month.strftime('%Y-%m')] = total_spend / active_customers

    average_spend_per_customer_same_days_series = pd.Series(average_spend_per_customer_same_days)

    # Calculate percentage change
    percentage_change = average_spend_per_customer_same_days_series.pct_change() * 100

    # Create the interactive line chart for monthly percentage change
    fig2 = go.Figure()

    # Add the average spend per customer (line)
    fig2.add_trace(go.Scatter(
        x=average_spend_per_customer_same_days_series.index,
        y=average_spend_per_customer_same_days_series,
        name='Average Spend per Customer (€)',
        mode='lines+markers+text',
        text=[f'{val:.2f}€' for val in average_spend_per_customer_same_days_series],
        textposition='top center',
        marker=dict(color='#0F8B8D', size=10),
        line=dict(color='#0F8B8D', width=2),
        textfont=dict(color='#0F8B8D')
    ))

    # Add percentage change as text labels (green for positive, red for negative)
    for i, month in enumerate(percentage_change.index):
        if i > 0:  # Skip the first element
            change = percentage_change.iloc[i]
            color = 'green' if change > 0 else 'red'
            fig2.add_annotation(
                x=month,
                y=average_spend_per_customer_same_days_series.iloc[i],
                text=f'{change:.2f}%',
                showarrow=True,
                arrowhead=2,
                font=dict(color=color)
            )

    # Update layout
    fig2.update_layout(
        title=f'Average Spend Per Customer (Same Time Frame as October MTD) for {region}',
        xaxis_title='Month',
        yaxis_title='Average Spend (€)',
        template='plotly_white'
    )

    # Show the figure
    fig2.show()

    # Part 3: Weekly Average Spend per Client and Percentage Change with ISO Week Numbers
    region_df['Week'] = region_df['Date de commande'].dt.isocalendar().week
    region_df['Year'] = region_df['Date de commande'].dt.isocalendar().year
    region_df['Week_Order'] = region_df['Year'].astype(str) + '-W' + region_df['Week'].astype(str)
    existing_weeks = region_df['Week_Order'].unique()

    # Calculate total spend and unique clients per week
    total_spend_per_week = region_df.groupby('Week_Order')['GMV euro'].sum()
    unique_clients_per_week = region_df.groupby('Week_Order')['Restaurant ID'].nunique()
    average_spend_per_client_per_week = total_spend_per_week / unique_clients_per_week

    # Create the interactive line chart for weekly average spend
    fig3 = go.Figure()

    # Add the average spend per client (line)
    fig3.add_trace(go.Scatter(
        x=average_spend_per_client_per_week.index,  # Use existing weeks
        y=average_spend_per_client_per_week,
        name='Weekly Average Spend per Client (€)',
        mode='lines+markers+text',
        text=[f'{val:.2f}€' for val in average_spend_per_client_per_week],
        textposition='top center',
        marker=dict(color='#0F8B8D', size=10),
        line=dict(color='#0F8B8D', width=2),
        textfont=dict(color='#0F8B8D')
    ))

    # Update layout
    fig3.update_layout(
        title=f'Weekly Average Spend Per Client (Linear) for {region}',
        xaxis_title='ISO Week Number',
        yaxis_title='Average Spend (€)',
        template='plotly_white'
    )

    # Show the figure
    fig3.show()

    # Calculate weekly percentage change
    percentage_change_weekly = average_spend_per_client_per_week.pct_change() * 100

    # Create the interactive line chart for weekly percentage change
    fig4 = go.Figure()

    # Add the percentage change (line)
    fig4.add_trace(go.Scatter(
        x=percentage_change_weekly.index,  # Use existing weeks
        y=percentage_change_weekly,
        name='Weekly Percentage Change in Average Spend',
        mode='lines+markers+text',
        text=[f'{val:.2f}%' for val in percentage_change_weekly],
        textposition='top center',
        marker=dict(color='#0F8B8D', size=10),
        line=dict(color='#0F8B8D', width=2),
        textfont=dict(color='#0F8B8D')
    ))

    # Add percentage change labels (green for positive, red for negative)
    for i, week in enumerate(percentage_change_weekly.index):
        if i > 0:  # Skip the first element
            change = percentage_change_weekly.iloc[i]
            color = 'green' if change > 0 else 'red'
            fig4.add_annotation(
                x=week,
                y=percentage_change_weekly.iloc[i],
                text=f'{change:.2f}%',
                showarrow=True,
                arrowhead=2,
                font=dict(color=color)
            )

    # Update layout
    fig4.update_layout(
        title=f'Weekly Percentage Change in Average Spend Per Client for {region}',
        xaxis_title='ISO Week Number',
        yaxis_title='Percentage Change (%)',
        template='plotly_white'
    )

    # Show the figure
    fig4.show()

    # Part 4: Waterfall Chart for MTD Changes in Average Spend Components (Frequency, Average Basket)

    # Step 1: Define the data for Last Month and This Month MTD
    last_month_name = region_df['Month_Year'].unique()[-2]  # Dynamically get the name of last month
    current_month_name = region_df['Month_Year'].unique()[-1]  # Dynamically get the name of current month

    data = {
        'Last Month': {
            'Frequency': region_df[region_df['Month_Year'] == last_month_name].shape[0] / unique_clients_per_month.loc[last_month_name],
            'Average Basket': total_spend_per_month.loc[last_month_name] / region_df[region_df['Month_Year'] == last_month_name].shape[0]
        },
        'This Month MTD': {
            'Frequency': region_df[region_df['Month_Year'] == current_month_name].shape[0] / unique_clients_per_month.loc[current_month_name],
            'Average Basket': total_spend_per_month.loc[current_month_name] / region_df[region_df['Month_Year'] == current_month_name].shape[0]
        }
    }

    # Step 2: Calculate last month's average spend (Orders * Frequency)
    last_month_avg_spend = average_spend_per_client_per_month.loc[last_month_name] if last_month_name in average_spend_per_client_per_month.index else 0
    current_month_avg_spend = average_spend_per_client_per_month.loc[current_month_name] if current_month_name in average_spend_per_client_per_month.index else 0

    # Step 3: Calculate percentage changes
    frequency_change_pct = (data['This Month MTD']['Frequency'] - data['Last Month']['Frequency']) / data['Last Month']['Frequency']
    average_basket_change_pct = (data['This Month MTD']['Average Basket'] - data['Last Month']['Average Basket']) / data['Last Month']['Average Basket']

    # Step 4: Calculate the contributions of each factor to the change in average spend
    frequency_contribution = frequency_change_pct * last_month_avg_spend
    average_basket_contribution = average_basket_change_pct * last_month_avg_spend

    # Step 5: Total change in average spend (calculated from the individual contributions)
    total_change = frequency_contribution + average_basket_contribution

    # Step 6: Create the Waterfall Chart
    fig5 = go.Figure(go.Waterfall(
        name="Average Spend Change",
        orientation="v",
        measure=["absolute", "relative", "relative", "absolute"],
        x=[f"{last_month_name} Avg Spend", "Frequency Contribution", "Average Basket Contribution", f"{current_month_name} Avg Spend"],
        textposition="outside",
        text=[f"{last_month_avg_spend:.2f}€", f"{frequency_contribution:.2f}€", f"{average_basket_contribution:.2f}€", f"{current_month_avg_spend:.2f}€"],
        y=[last_month_avg_spend, frequency_contribution, average_basket_contribution, current_month_avg_spend],
        connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))

    # Add chart title and labels
    fig5.update_layout(
        title=f"Waterfall Chart of Average Spend Change for {region} ({last_month_name} to {current_month_name})",
        xaxis_title="Components",
        yaxis_title="Contribution (€)",
        waterfallgap=0.3
    )

    # Show the chart
    fig5.show()


# @title
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Path to the combined data (for .xlsx file)
combined_data_path = '/content/drive/Shareddrives/Commandes BO/PDATA.xlsx'

def load_combined_data():
    print("Loading combined purchase data from Excel...")
    combined_purchase_data = pd.read_excel(combined_data_path, engine='openpyxl')

    # Ensure the 'Date' column is in datetime format and create 'month_year' column
    combined_purchase_data['Date'] = pd.to_datetime(combined_purchase_data['Date'], errors='coerce')
    combined_purchase_data = combined_purchase_data.dropna(subset=['Date'])  # Drop rows where 'Date' could not be converted
    combined_purchase_data['month_year'] = combined_purchase_data['Date'].dt.to_period('M')  # Create 'month_year' column

    print(f"Combined data shape: {combined_purchase_data.shape}")
    print("Unique regions: ", combined_purchase_data['region'].unique())  # Add this to ensure all regions are present
    return combined_purchase_data

# Function to filter Mono-Category clients only
def filter_mono_category_clients(df):
    category_counts = df.groupby(['Restaurant_id', 'month_year'])['Product Category'].nunique().reset_index()
    mono_category_clients = category_counts[category_counts['Product Category'] == 1]

    # Keep only mono-category clients in the original dataset
    mono_clients = pd.merge(df, mono_category_clients[['Restaurant_id', 'month_year']], on=['Restaurant_id', 'month_year'], how='inner')
    return mono_clients

# Function to calculate Mono and Multi-Category clients per month
def mono_multi_category_clients(df):
    category_counts = df.groupby(['Restaurant_id', 'month_year'])['Product Category'].nunique().reset_index()
    mono_category_clients = category_counts[category_counts['Product Category'] == 1]
    multi_category_clients = category_counts[category_counts['Product Category'] > 1]

    mono_per_month = mono_category_clients.groupby('month_year')['Restaurant_id'].nunique().reset_index()
    multi_per_month = multi_category_clients.groupby('month_year')['Restaurant_id'].nunique().reset_index()

    combined = pd.merge(mono_per_month, multi_per_month, on='month_year', how='outer', suffixes=('_mono', '_multi')).fillna(0)
    combined['total_clients'] = combined['Restaurant_id_mono'] + combined['Restaurant_id_multi']
    combined['mono_percentage'] = (combined['Restaurant_id_mono'] / combined['total_clients']) * 100
    combined['multi_percentage'] = (combined['Restaurant_id_multi'] / combined['total_clients']) * 100
    return combined

# Function to plot Mono and Multi-Category Clients per Month (Percentage)
def plot_mono_multi_clients_percentage(combined):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined['month_year'].astype(str),
        y=combined['mono_percentage'],
        name='Mono-category',
        marker=dict(color='#1F271B'),
        text=combined['mono_percentage'].round(1),
        textposition='inside',
    ))
    fig.add_trace(go.Bar(
        x=combined['month_year'].astype(str),
        y=combined['multi_percentage'],
        name='Multi-category',
        marker=dict(color='#145C9E'),
        text=combined['multi_percentage'].round(1),
        textposition='inside',
    ))

    fig.update_layout(
        barmode='stack',
        title='Mono and Multi-Category Clients per Month (Percentage)',
        xaxis_title='Month-Year',
        yaxis_title='Percentage of Clients (%)',
        yaxis=dict(range=[0, 100]),  # Make sure the graph stays within 100%
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )
    fig.show()

# Function to plot Mono and Multi-Category Clients per Month (Count)
def plot_mono_multi_clients_count(combined):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined['month_year'].astype(str),
        y=combined['Restaurant_id_mono'],
        name='Mono-category',
        marker=dict(color='#1F271B'),
        text=combined['Restaurant_id_mono'].round(1),
        textposition='inside',
    ))
    fig.add_trace(go.Bar(
        x=combined['month_year'].astype(str),
        y=combined['Restaurant_id_multi'],
        name='Multi-category',
        marker=dict(color='#145C9E'),
        text=combined['Restaurant_id_multi'].round(1),
        textposition='inside',
    ))

    fig.update_layout(
        barmode='stack',
        title='Mono and Multi-Category Clients per Month (Count)',
        xaxis_title='Month-Year',
        yaxis_title='Number of Clients',
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )
    fig.show()

# Function to calculate Month-over-Month changes
def calculate_month_over_month_change(combined):
    combined['mono_change'] = combined['Restaurant_id_mono'].pct_change() * 100
    combined['multi_change'] = combined['Restaurant_id_multi'].pct_change() * 100
    return combined

# Function to plot the line graph of Month-over-Month changes
def plot_mono_multi_clients_change(combined):
    fig = go.Figure()

    # Plot Mono-Category Change
    fig.add_trace(go.Scatter(
        x=combined['month_year'].astype(str),
        y=combined['mono_change'],
        mode='lines',
        name='Mono-category Change',
        line=dict(color='rgb(67,67,67)', width=2),
        connectgaps=True
    ))

    # Plot Multi-Category Change
    fig.add_trace(go.Scatter(
        x=combined['month_year'].astype(str),
        y=combined['multi_change'],
        mode='lines',
        name='Multi-category Change',
        line=dict(color='rgb(49,130,189)', width=4),
        connectgaps=True
    ))

    fig.update_layout(
        title='Month-over-Month Change in Mono and Multi-Category Clients',
        xaxis_title='Month-Year',
        yaxis_title='Percentage Change (%)',
        yaxis=dict(range=[-100, 100]),  # Limit the y-axis
        width=1000,
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )

    fig.show()

# Function to calculate Month-to-Date (MTD) changes
def calculate_mtd_change(df):
    current_day = datetime.now().day
    df['day'] = df['Date'].dt.day

    # Filter all months to have only up to the current day
    df_mtd = df[df['day'] <= current_day]

    # Group and calculate Mono and Multi clients for each truncated month
    mtd_mono_clients = df_mtd.groupby(['month_year', 'Restaurant_id'])['Product Category'].nunique().reset_index()
    mtd_mono_clients['client_type'] = mtd_mono_clients['Product Category'].apply(lambda x: 'Mono' if x == 1 else 'Multi')

    mtd_mono_per_month = mtd_mono_clients[mtd_mono_clients['client_type'] == 'Mono'].groupby('month_year')['Restaurant_id'].nunique().reset_index(name='Restaurant_id_mono')
    mtd_multi_per_month = mtd_mono_clients[mtd_mono_clients['client_type'] == 'Multi'].groupby('month_year')['Restaurant_id'].nunique().reset_index(name='Restaurant_id_multi')

    combined_mtd = pd.merge(mtd_mono_per_month, mtd_multi_per_month, on='month_year', how='outer').fillna(0)
    combined_mtd['total_clients'] = combined_mtd['Restaurant_id_mono'] + combined_mtd['Restaurant_id_multi']
    combined_mtd['mono_percentage'] = (combined_mtd['Restaurant_id_mono'] / combined_mtd['total_clients']) * 100
    combined_mtd['multi_percentage'] = (combined_mtd['Restaurant_id_multi'] / combined_mtd['total_clients']) * 100

    # Calculate MTD percentage change
    combined_mtd['mono_change'] = combined_mtd['Restaurant_id_mono'].pct_change() * 100
    combined_mtd['multi_change'] = combined_mtd['Restaurant_id_multi'].pct_change() * 100

    return combined_mtd

# Function to plot Month-to-Date (MTD) changes
def plot_mtd_mono_multi_clients_change(df):
    combined_mtd = calculate_mtd_change(df)

    fig = go.Figure()

    # Plot MTD Mono-Category Change
    fig.add_trace(go.Scatter(
        x=combined_mtd['month_year'].astype(str),
        y=combined_mtd['mono_change'],
        mode='lines',
        name='MTD Mono-category Change',
        line=dict(color='rgb(67,67,67)', width=2),
        connectgaps=True
    ))

    # Plot MTD Multi-Category Change
    fig.add_trace(go.Scatter(
        x=combined_mtd['month_year'].astype(str),
        y=combined_mtd['multi_change'],
        mode='lines',
        name='MTD Multi-category Change',
        line=dict(color='rgb(49,130,189)', width=4),
        connectgaps=True
    ))

    fig.update_layout(
        title='Month-to-Date Change in Mono and Multi-Category Clients',
        xaxis_title='Month-Year',
        yaxis_title='MTD Percentage Change (%)',
        yaxis=dict(range=[-100, 100]),  # Limit the y-axis
        width=1000,
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )

    fig.show()

# Function to calculate Mono Clients per Product Category (Each Bar = 100%)
def mono_clients_per_product(df):
    mono_clients = filter_mono_category_clients(df)

    mono_category_counts = mono_clients.groupby(['month_year', 'Product Category'])['Restaurant_id'].nunique().reset_index()
    total_mono_per_month = mono_clients.groupby('month_year')['Restaurant_id'].nunique().reset_index()

    # Calculate percentages relative to the total mono clients per month
    mono_category_counts = pd.merge(mono_category_counts, total_mono_per_month, on='month_year', suffixes=('_category', '_total'))
    mono_category_counts['percentage'] = (mono_category_counts['Restaurant_id_category'] / mono_category_counts['Restaurant_id_total']) * 100

    pivot_data = mono_category_counts.pivot(index='month_year', columns='Product Category', values='percentage').fillna(0)

    return pivot_data, mono_category_counts

# Function to plot Mono Clients per Product Category (Each Bar = 100%)
def plot_mono_clients_per_product(df):
    pivot_data, mono_category_counts = mono_clients_per_product(df)

    fig = go.Figure()
    colors = ['#1F271B', '#145C9E', '#CBB9A8'] + ['#DAD2D8', '#EC9A29', '#A8201A', '#FB9F89', '#0F8B8D']

    for idx, category in enumerate(pivot_data.columns):
        fig.add_trace(go.Bar(
            x=pivot_data.index.astype(str),
            y=pivot_data[category],
            name=category,
            marker=dict(color=colors[idx % len(colors)]),
            text=pivot_data[category].round(1),
            textposition='inside',
        ))

    fig.update_layout(
        barmode='stack',
        title='Mono Clients per Product Category (Percentage of Mono Clients per Month)',
        xaxis_title='Month-Year',
        yaxis_title='Percentage of Mono-Category Clients (%)',
        yaxis=dict(range=[0, 100]),  # Limit y-axis to 100%
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )
    fig.show()

    # Table for the data used in the graph
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=["Month-Year", "Product Category", "Restaurant_id (Category)", "Total Mono Clients", "Percentage"],
                    fill_color='#DAD2D8',  # Timberwolf background for table
                    align='left'),
        cells=dict(values=[mono_category_counts['month_year'].astype(str),
                           mono_category_counts['Product Category'],
                           mono_category_counts['Restaurant_id_category'],
                           mono_category_counts['Restaurant_id_total'],
                           mono_category_counts['percentage'].round(2)],
                   fill_color='lavender',
                   align='left'))
    ])
    table_fig.show()

# Function to plot Mono Clients per Product Category for the Current Month (Each Bar = 100%)
def mono_clients_per_product_current_month(df):
    current_month = datetime.now().strftime("%Y-%m")
    mono_clients = filter_mono_category_clients(df)

    df_current_month = mono_clients[mono_clients['month_year'].astype(str) == current_month]

    mono_category_counts = df_current_month.groupby(['region', 'Product Category'])['Restaurant_id'].nunique().reset_index()
    total_mono_per_region = df_current_month.groupby('region')['Restaurant_id'].nunique().reset_index()

    # Calculate percentages relative to the total mono clients per region
    mono_category_counts = pd.merge(mono_category_counts, total_mono_per_region, on='region', suffixes=('_category', '_total'))
    mono_category_counts['percentage'] = (mono_category_counts['Restaurant_id_category'] / mono_category_counts['Restaurant_id_total']) * 100

    pivot_data = mono_category_counts.pivot(index='region', columns='Product Category', values='percentage').fillna(0)

    return pivot_data, mono_category_counts

def plot_mono_clients_per_product_current_month(df):
    pivot_data, mono_category_counts = mono_clients_per_product_current_month(df)

    fig = go.Figure()
    colors = ['#1F271B', '#145C9E', '#CBB9A8'] + ['#DAD2D8', '#EC9A29', '#A8201A', '#FB9F89', '#0F8B8D']

    for idx, category in enumerate(pivot_data.columns):
        fig.add_trace(go.Bar(
            x=pivot_data.index.astype(str),
            y=pivot_data[category],
            name=category,
            marker=dict(color=colors[idx % len(colors)]),
            text=pivot_data[category].round(1),
            textposition='inside',
        ))

    fig.update_layout(
        barmode='stack',
        title=f'Mono Clients per Product Category (Current Month: {datetime.now().strftime("%Y-%m")})',
        xaxis_title='Region',
        yaxis_title='Percentage of Mono-Category Clients (%)',
        yaxis=dict(range=[0, 100]),  # Limit y-axis to 100%
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )
    fig.show()

    # Table for the data used in the graph
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=["Region", "Product Category", "Restaurant_id (Category)", "Total Mono Clients", "Percentage"],
                    fill_color='#DAD2D8',  # Timberwolf background for table
                    align='left'),
        cells=dict(values=[mono_category_counts['region'],
                           mono_category_counts['Product Category'],
                           mono_category_counts['Restaurant_id_category'],
                           mono_category_counts['Restaurant_id_total'],
                           mono_category_counts['percentage'].round(2)],
                   fill_color='lavender',
                   align='left'))
    ])
    table_fig.show()

# Function to plot Percentage of Mono-Category Clients by Region for the Current Month
def plot_mono_by_region_current_month(df):
    current_month = datetime.now().strftime("%Y-%m")
    df_current_month = df[df['month_year'].astype(str) == current_month]

    # Debug: Check if all regions are in the current month's data
    print("Regions in current month data:", df_current_month['region'].unique())

    # Get a list of all possible regions or countries (use country if region is not properly defined)
    all_regions = df_current_month['region'].unique()

    # Group the data to get mono-category clients and total clients by region
    region_mono_counts = df_current_month.groupby(['Restaurant_id', 'region'])['Product Category'].nunique().reset_index()
    mono_clients = region_mono_counts[region_mono_counts['Product Category'] == 1].groupby('region')['Restaurant_id'].nunique().reset_index()
    total_clients = df_current_month.groupby('region')['Restaurant_id'].nunique().reset_index()

    # Merge mono clients and total clients to calculate percentage
    region_mono_percentage = pd.merge(mono_clients, total_clients, on='region', suffixes=('_mono', '_total'), how='right')

    # Ensure all regions are included, even if missing data (fill with 0 if needed)
    region_mono_percentage = region_mono_percentage.set_index('region').reindex(all_regions).fillna(0).reset_index()
    region_mono_percentage['percentage'] = (region_mono_percentage['Restaurant_id_mono'] / region_mono_percentage['Restaurant_id_total']) * 100

    # Plot the data with color assignment
    colors = ['#1F271B', '#145C9E', '#CBB9A8'] + ['#DAD2D8', '#EC9A29', '#A8201A', '#FB9F89', '#0F8B8D']
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=region_mono_percentage['region'],
        y=region_mono_percentage['percentage'],
        marker=dict(color=colors[:len(region_mono_percentage)]),  # Use appropriate colors
        text=region_mono_percentage['percentage'].round(1),
        textposition='inside'
    ))

    fig.update_layout(
        title=f'Percentage of Mono-Category Clients by Region (Current Month: {current_month})',
        xaxis_title='Region',
        yaxis_title='Percentage of Mono-Category Clients (%)',
        yaxis=dict(range=[0, 100]),  # Limit y-axis to 100%
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )
    fig.show()

    # Table for the data used in the graph
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=["Region", "Restaurant_id_mono", "Restaurant_id_total", "Percentage"],
                    fill_color='#DAD2D8',  # Timberwolf background for table
                    align='left'),
        cells=dict(values=[region_mono_percentage['region'],
                           region_mono_percentage['Restaurant_id_mono'],
                           region_mono_percentage['Restaurant_id_total'],
                           region_mono_percentage['percentage'].round(2)],
                   fill_color='lavender',
                   align='left'))
    ])
    table_fig.show()

# Main function to load data and generate the graphs and tables
def main():
    # Load the combined purchase data from the saved file
    combined_purchase_data = load_combined_data()

    # Generate Mono and Multi-Category Clients per Month (Percentage)
    combined = mono_multi_category_clients(combined_purchase_data)
    plot_mono_multi_clients_percentage(combined)

    # Generate Mono and Multi-Category Clients per Month (Count)
    plot_mono_multi_clients_count(combined)

    # Generate and display the line graph for month-over-month change (added after second graph)
    combined = calculate_month_over_month_change(combined)
    plot_mono_multi_clients_change(combined)

    # Generate and display the MTD (Month-to-Date) line graph for Mono and Multi clients
    plot_mtd_mono_multi_clients_change(combined_purchase_data)

    # Generate Mono Clients per Product Category (Percentage of Mono Clients per Month)
    plot_mono_clients_per_product(combined_purchase_data)

    # Generate Mono Clients per Product Category for the Current Month (Percentage of Mono Clients per Region)
    plot_mono_clients_per_product_current_month(combined_purchase_data)

    # Generate Percentage of Mono-Category Clients by Region for the Current Month
    plot_mono_by_region_current_month(combined_purchase_data)

# Execute the main function
if __name__ == "__main__":
    main()

# @title
import pandas as pd
import plotly.graph_objects as go
import warnings
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output

# Suppress warnings about missing styles in Excel files
warnings.simplefilter(action='ignore', category=UserWarning)

# Load data from the new file
df = pd.read_excel('/content/drive/Shareddrives/Commandes BO/PDATA.xlsx', engine='openpyxl')

# Drop rows with NaN values in essential columns (like 'Date', 'Product Category', 'Restaurant_id', 'Region')
df = df.dropna(subset=['Date', 'Product Category', 'Restaurant_id', 'region'])

# Convert date columns to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Get the current date and calculate the number of days since the start of the current month
today = datetime.today()
current_year = today.year
current_month = today.month
current_day = today.day

# Define the start date for the current month
current_month_start = datetime(current_year, current_month, 1)

# Calculate the equivalent start and end dates for the previous month
previous_month = current_month - 1 if current_month > 1 else 12
previous_year = current_year if current_month > 1 else current_year - 1
previous_month_start = datetime(previous_year, previous_month, 1)

# Calculate the last day of the equivalent period in the previous month
previous_month_end = previous_month_start + timedelta(days=current_day - 1)

# Filter data dynamically for both periods
current_month_df = df[(df['Date'] >= current_month_start) & (df['Date'] <= today)]
previous_month_df = df[(df['Date'] >= previous_month_start) & (df['Date'] <= previous_month_end)]

# Get the names of the current and previous months for naming the DataFrames
current_month_name = today.strftime("%B")
previous_month_name = previous_month_start.strftime("%B")

# Define the unique regions in the dataset for the filter
unique_regions = df['region'].unique().tolist()

# Create a Dash app
app = Dash(__name__)

# Define the layout for the Dash app
app.layout = html.Div([
    html.H1(f"Client Movement from {previous_month_name} to {current_month_name} 2024"),

    # Dropdown for selecting regions
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': region, 'value': region} for region in unique_regions] + [{'label': 'Global', 'value': 'Global'}],
        multi=True,
        placeholder="Select Region(s) to filter",
        value=['Global']  # Default is Global view
    ),

    # Sankey Diagram
    dcc.Graph(id='sankey-diagram')
])

# Function to categorize clients based on the number of unique categories
def categorize_client_categories(n_categories):
    if n_categories == 1:
        return 'Mono-Category'
    elif n_categories >= 2:
        return 'Multi-Category'

# Function to filter data based on selected region(s)
def filter_by_region(selected_regions):
    if not selected_regions or 'Global' in selected_regions:
        return current_month_df, previous_month_df  # Return global data if nothing is selected or 'Global' is selected
    else:
        current_filtered = current_month_df[current_month_df['region'].isin(selected_regions)]
        previous_filtered = previous_month_df[previous_month_df['region'].isin(selected_regions)]
        return current_filtered, previous_filtered

# Function to create the Sankey diagram based on filtered data
def create_sankey(selected_regions):
    # Filter the data
    current_filtered, previous_filtered = filter_by_region(selected_regions)

    # Calculate the unique number of product categories per Restaurant ID for each period
    current_categories = current_filtered.groupby('Restaurant_id')['Product Category'].nunique()
    previous_categories = previous_filtered.groupby('Restaurant_id')['Product Category'].nunique()

    # Categorize clients based on the number of unique categories
    current_filtered['Category Group'] = current_filtered['Restaurant_id'].map(current_categories).apply(categorize_client_categories)
    previous_filtered['Category Group'] = previous_filtered['Restaurant_id'].map(previous_categories).apply(categorize_client_categories)

    # Track transitions from previous month to current month, including churned clients
    transitions = {
        'Mono-Category': {'Mono-Category': 0, 'Multi-Category': 0, 'Churned': 0},
        'Multi-Category': {'Mono-Category': 0, 'Multi-Category': 0, 'Churned': 0},
    }

    previous_category_group = previous_filtered.groupby('Restaurant_id')['Category Group'].first().to_dict()
    current_category_group = current_filtered.groupby('Restaurant_id')['Category Group'].first().to_dict()

    for restaurant_id in previous_category_group:
        if restaurant_id in current_category_group:
            transitions[previous_category_group[restaurant_id]][current_category_group[restaurant_id]] += 1
        else:
            transitions[previous_category_group[restaurant_id]]['Churned'] += 1

    # Prepare data for the Sankey diagram
    labels = [f"{previous_month_name} - Mono-Category", f"{previous_month_name} - Multi-Category",
              f"{current_month_name} - Mono-Category", f"{current_month_name} - Multi-Category", f"{current_month_name} - Churned"]

    source = []
    target = []
    values = []
    colors = []

    for i, prev_cat in enumerate(['Mono-Category', 'Multi-Category']):
        for j, curr_cat in enumerate(['Mono-Category', 'Multi-Category', 'Churned']):
            if transitions[prev_cat][curr_cat] > 0:
                source.append(i)
                target.append(j + 2)
                values.append(transitions[prev_cat][curr_cat])

                # Color-coding transitions
                if j > i and curr_cat != 'Churned':  # Upward movement
                    colors.append('lightgreen')
                elif j < i or curr_cat == 'Churned':  # Downward movement or Churned
                    colors.append('lightcoral')
                else:  # No change
                    colors.append('lightgray')

    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=colors
        )
    ))

    fig.update_layout(title_text=f"Client Movement from {previous_month_name} to {current_month_name} 2024", font_size=10)
    return fig

# Define callback to update the Sankey diagram when the dropdown selection changes
@app.callback(
    Output('sankey-diagram', 'figure'),
    [Input('region-dropdown', 'value')]
)
def update_sankey(selected_regions):
    return create_sankey(selected_regions)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
