import matplotlib.pyplot as plt
import pandas as pd
import re
import io
import math
import os
import numpy as np

# Display execution metadata
print(f"Script executed by Lachy-Dauth at 2025-04-08 03:30:14 UTC")

def extract_activity_log_data(file_path):
    """Extract only the activity log data from a mixed log file."""
    activity_data_lines = []
    in_activity_section = False
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                # Check if this is the header line of the activity log
                if "day;timestamp;product;" in line:
                    in_activity_section = True
                    activity_data_lines.append(line.strip())
                    continue
                    
                # If we're in the activity section and line has the correct format
                if in_activity_section and re.match(r'^-?\d+;\d+;[A-Z_]+;', line):
                    activity_data_lines.append(line.strip())
                elif in_activity_section and line.strip() == "":
                    # Empty line might indicate end of section
                    continue
                elif in_activity_section:
                    # If we encounter a different format while in activity section, we're done
                    in_activity_section = False
                    
        print(f"Extracted {len(activity_data_lines)} activity log lines")
        return activity_data_lines
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Extract activity data from the log file
activity_lines = extract_activity_log_data("data.log")

if not activity_lines:
    # If extraction failed, use the data from the previous message as fallback
    print("Using fallback data from previous message")
    activity_data = """day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
-1;0;RAINFOREST_RESIN;10002;1;9996;2;9995;29;10004;2;10005;29;;;10003.0;0.0
-1;0;KELP;2028;1;2026;2;2025;29;2029;31;;;;;2028.5;0.0
-1;100;KELP;2025;24;;;;;2028;2;2029;22;;;2026.5;0.0
-1;100;RAINFOREST_RESIN;9996;2;9995;22;;;10004;2;10005;22;;;10000.0;0.0
-1;200;RAINFOREST_RESIN;9995;20;;;;;10005;20;;;;;10000.0;0.0
-1;200;KELP;2025;22;;;;;2028;20;;;;;2026.5;0.0
-1;300;KELP;2025;31;;;;;2028;2;2029;29;;;2026.5;0.0
-1;300;RAINFOREST_RESIN;9996;2;9995;29;;;10004;2;10005;29;;;10000.0;0.0
-1;400;RAINFOREST_RESIN;9996;2;9995;25;;;10004;2;10005;25;;;10000.0;0.0
-1;400;KELP;2025;27;;;;;2028;27;;;;;2026.5;0.0
-1;500;KELP;2025;30;;;;;2028;30;;;;;2026.5;0.0
-1;500;RAINFOREST_RESIN;9995;30;;;;;10002;4;10005;30;;;9998.5;0.0"""
    df = pd.read_csv(io.StringIO(activity_data), sep=";")
else:
    # Convert extracted lines to a DataFrame
    activity_data = "\n".join(activity_lines)
    df = pd.read_csv(io.StringIO(activity_data), sep=";")

print(f"Processing data with {len(df)} total rows")
print(f"Columns in the data: {df.columns.tolist()}")

# Create an output directory for CSV files if it doesn't exist
csv_dir = "commodity_prices"
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
    print(f"Created directory {csv_dir} for CSV exports")

# Get unique products in the data
unique_products = df['product'].unique()
print(f"Found {len(unique_products)} unique products: {', '.join(unique_products)}")

# Create a dictionary to store product-specific dataframes
product_dfs = {}

# Filter data by product and export CSV files
for product in unique_products:
    product_dfs[product] = df[df['product'] == product].sort_values('timestamp')
    print(f"Found {len(product_dfs[product])} {product} records")
    
    # Export price data to CSV
    csv_filename = os.path.join(csv_dir, f"{product}_prices.csv")
    
    # Select only the columns we want to export
    price_data = product_dfs[product][['timestamp', 'mid_price', 'bid_price_1', 'ask_price_1']]
    price_data.to_csv(csv_filename, index=False)
    print(f"Exported price data for {product} to {csv_filename}")

# Determine how to layout the plots based on number of products
def get_subplot_layout(n):
    """Determine a reasonable subplot layout for n plots"""
    rows = max(1, math.ceil(n / 2))
    cols = min(n, 2)
    return rows, cols

# Function to plot price trends for a set of products
def plot_price_trends(product_dfs):
    rows, cols = get_subplot_layout(len(product_dfs))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle('Price Trends Over Time', fontsize=16)
    
    # Make sure axes is always a 2D array
    if len(product_dfs) == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    i, j = 0, 0
    for product, product_df in product_dfs.items():
        ax = axes[i][j]
        ax.plot(product_df['timestamp'], product_df['mid_price'], 'b-o', label='Mid Price')
        ax.plot(product_df['timestamp'], product_df['bid_price_1'], 'g--^', label='Top Bid')
        ax.plot(product_df['timestamp'], product_df['ask_price_1'], 'r--v', label='Top Ask')
        ax.set_title(f'{product} Price Trends')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price (SEASHELLS)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations for significant price changes
        if len(product_df) > 1:
            product_df['price_change'] = product_df['mid_price'].diff().abs()
            max_change_idx = product_df['price_change'].idxmax()
            if not pd.isna(max_change_idx):
                max_change_row = product_df.loc[max_change_idx]
                ax.annotate(f'Price Shift: {max_change_row["price_change"]:.2f}', 
                         xy=(max_change_row['timestamp'], max_change_row['mid_price']), 
                         xytext=(max_change_row['timestamp']-50, max_change_row['mid_price']+1),
                         arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Update position for next plot
        j += 1
        if j >= cols:
            j = 0
            i += 1
    
    # Hide unused subplots
    for i_pad in range(i, rows):
        for j_pad in range((0 if i_pad > i else j), cols):
            fig.delaxes(axes[i_pad][j_pad])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('market_price_analysis.png', dpi=300)
    print("Saved price analysis chart to market_price_analysis.png")

# Function to plot bid-ask spreads
def plot_bid_ask_spreads(product_dfs):
    rows, cols = get_subplot_layout(len(product_dfs))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle('Bid-Ask Spread Over Time', fontsize=16)
    
    # Make sure axes is always a 2D array
    if len(product_dfs) == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    i, j = 0, 0
    for product, product_df in product_dfs.items():
        ax = axes[i][j]
        product_df['spread'] = product_df['ask_price_1'] - product_df['bid_price_1']
        ax.plot(product_df['timestamp'], product_df['spread'], 'm-o')
        ax.set_title(f'{product} Bid-Ask Spread')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Spread (SEASHELLS)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Update position for next plot
        j += 1
        if j >= cols:
            j = 0
            i += 1
    
    # Hide unused subplots
    for i_pad in range(i, rows):
        for j_pad in range((0 if i_pad > i else j), cols):
            fig.delaxes(axes[i_pad][j_pad])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('spread_analysis.png', dpi=300)
    print("Saved spread analysis chart to spread_analysis.png")

# Function to plot volumes
def plot_volumes(product_dfs):
    rows, cols = get_subplot_layout(len(product_dfs))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle('Volume Analysis Over Time', fontsize=16)
    
    # Make sure axes is always a 2D array
    if len(product_dfs) == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    i, j = 0, 0
    for product, product_df in product_dfs.items():
        ax = axes[i][j]
        ax.bar(product_df['timestamp'] - 2, product_df['bid_volume_1'], width=4, color='green', alpha=0.6, label='Top Bid Volume')
        ax.bar(product_df['timestamp'] + 2, product_df['ask_volume_1'], width=4, color='red', alpha=0.6, label='Top Ask Volume')
        ax.set_title(f'{product} Volume')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Update position for next plot
        j += 1
        if j >= cols:
            j = 0
            i += 1
    
    # Hide unused subplots
    for i_pad in range(i, rows):
        for j_pad in range((0 if i_pad > i else j), cols):
            fig.delaxes(axes[i_pad][j_pad])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('volume_analysis.png', dpi=300)
    print("Saved volume analysis chart to volume_analysis.png")

# Create the three types of visualizations
plot_price_trends(product_dfs)
plot_bid_ask_spreads(product_dfs)
plot_volumes(product_dfs)

print("Analysis complete!")