import matplotlib.pyplot as plt
import pandas as pd
import re
import io

# Display execution metadata
print(f"Script executed by Lachy-Dauth at 2025-04-06 05:30:13 UTC")

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

# Filter to only the relevant products
kelp_df = df[df['product'] == 'KELP'].sort_values('timestamp')
resin_df = df[df['product'] == 'RAINFOREST_RESIN'].sort_values('timestamp')

print(f"Found {len(kelp_df)} KELP records and {len(resin_df)} RAINFOREST_RESIN records")

# Create a figure with two subplots (one for each product)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Price Trends Over Time', fontsize=16)

# Plot KELP prices
ax1.plot(kelp_df['timestamp'], kelp_df['mid_price'], 'b-o', label='Mid Price')
ax1.plot(kelp_df['timestamp'], kelp_df['bid_price_1'], 'g--^', label='Top Bid')
ax1.plot(kelp_df['timestamp'], kelp_df['ask_price_1'], 'r--v', label='Top Ask')
ax1.set_title('KELP Price Trends')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Price (SEASHELLS)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot RAINFOREST_RESIN prices
ax2.plot(resin_df['timestamp'], resin_df['mid_price'], 'b-o', label='Mid Price')
ax2.plot(resin_df['timestamp'], resin_df['bid_price_1'], 'g--^', label='Top Bid')
ax2.plot(resin_df['timestamp'], resin_df['ask_price_1'], 'r--v', label='Top Ask')
ax2.set_title('RAINFOREST_RESIN Price Trends')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Price (SEASHELLS)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add annotations for significant price changes
if len(kelp_df) > 1:
    kelp_df['price_change'] = kelp_df['mid_price'].diff().abs()
    max_change_idx = kelp_df['price_change'].idxmax()
    if not pd.isna(max_change_idx):
        max_change_row = kelp_df.loc[max_change_idx]
        ax1.annotate(f'Price Shift: {max_change_row["price_change"]:.2f}', 
                     xy=(max_change_row['timestamp'], max_change_row['mid_price']), 
                     xytext=(max_change_row['timestamp']-50, max_change_row['mid_price']+1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

if len(resin_df) > 1:
    resin_df['price_change'] = resin_df['mid_price'].diff().abs()
    max_change_idx = resin_df['price_change'].idxmax()
    if not pd.isna(max_change_idx):
        max_change_row = resin_df.loc[max_change_idx]
        ax2.annotate(f'Price Shift: {max_change_row["price_change"]:.2f}', 
                     xy=(max_change_row['timestamp'], max_change_row['mid_price']), 
                     xytext=(max_change_row['timestamp']-50, max_change_row['mid_price']+5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('market_price_analysis.png', dpi=300)
print("Saved price analysis chart to market_price_analysis.png")

# Create a second figure to show the bid-ask spread over time
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
fig2.suptitle('Bid-Ask Spread Over Time', fontsize=16)

# Calculate spreads
kelp_df['spread'] = kelp_df['ask_price_1'] - kelp_df['bid_price_1']
resin_df['spread'] = resin_df['ask_price_1'] - resin_df['bid_price_1']

# Plot spreads
ax3.plot(kelp_df['timestamp'], kelp_df['spread'], 'm-o', label='KELP Spread')
ax3.set_title('KELP Bid-Ask Spread')
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Spread (SEASHELLS)')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)

ax4.plot(resin_df['timestamp'], resin_df['spread'], 'c-o', label='RAINFOREST_RESIN Spread')
ax4.set_title('RAINFOREST_RESIN Bid-Ask Spread')
ax4.set_xlabel('Timestamp')
ax4.set_ylabel('Spread (SEASHELLS)')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('spread_analysis.png', dpi=300)
print("Saved spread analysis chart to spread_analysis.png")

# Add volume analysis
fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 8))
fig3.suptitle('Volume Analysis Over Time', fontsize=16)

# Plot volumes for KELP
ax5.bar(kelp_df['timestamp'] - 2, kelp_df['bid_volume_1'], width=4, color='green', alpha=0.6, label='Top Bid Volume')
ax5.bar(kelp_df['timestamp'] + 2, kelp_df['ask_volume_1'], width=4, color='red', alpha=0.6, label='Top Ask Volume')
ax5.set_title('KELP Volume')
ax5.set_xlabel('Timestamp')
ax5.set_ylabel('Volume')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot volumes for RAINFOREST_RESIN
ax6.bar(resin_df['timestamp'] - 2, resin_df['bid_volume_1'], width=4, color='green', alpha=0.6, label='Top Bid Volume')
ax6.bar(resin_df['timestamp'] + 2, resin_df['ask_volume_1'], width=4, color='red', alpha=0.6, label='Top Ask Volume')
ax6.set_title('RAINFOREST_RESIN Volume')
ax6.set_xlabel('Timestamp')
ax6.set_ylabel('Volume')
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('volume_analysis.png', dpi=300)
print("Saved volume analysis chart to volume_analysis.png")

print("Analysis complete!")