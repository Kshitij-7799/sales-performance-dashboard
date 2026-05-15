# ============================================================
# Sales Performance Dashboard
# Author: Kshitij Kumar
# Description: Analyze 3 years of sales data and build visual dashboard
# Tools: Python, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Generate / Load Sales Data
# ============================================================
# If you have real data: df = pd.read_csv('sales_data.csv')
# Below we generate realistic sample data for demonstration

np.random.seed(42)
n = 100000

regions = ['North', 'South', 'East', 'West']
products = ['Laptop', 'Mobile', 'Tablet', 'Accessories', 'Headphones']
sales_reps = [f'Rep_{i}' for i in range(1, 21)]

dates = pd.date_range(start='2022-01-01', end='2024-12-31', periods=n)

df = pd.DataFrame({
    'date': dates,
    'region': np.random.choice(regions, n),
    'product': np.random.choice(products, n),
    'sales_rep': np.random.choice(sales_reps, n),
    'units_sold': np.random.randint(1, 20, n),
    'unit_price': np.random.choice([15000, 25000, 8000, 1500, 3000], n),
})

df['revenue'] = df['units_sold'] * df['unit_price']
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['year_month'] = df['date'].dt.to_period('M')

print("Dataset Shape:", df.shape)
print("\nSample Data:")
print(df.head())
print("\nTotal Revenue: ₹", df['revenue'].sum():,.0f)

# ============================================================
# STEP 2: Data Cleaning
# ============================================================

print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# ============================================================
# STEP 3: EDA & Dashboard
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Sales Performance Dashboard — 2022 to 2024', fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Chart 1: Monthly Revenue Trend ---
ax1 = fig.add_subplot(gs[0, :])
monthly = df.groupby('year_month')['revenue'].sum().reset_index()
monthly['year_month_str'] = monthly['year_month'].astype(str)
ax1.plot(monthly['year_month_str'], monthly['revenue'] / 1e6, color='steelblue', linewidth=2, marker='o', markersize=3)
ax1.fill_between(monthly['year_month_str'], monthly['revenue'] / 1e6, alpha=0.15, color='steelblue')
ax1.set_title('Monthly Revenue Trend (₹ Millions)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (₹M)')
step = max(1, len(monthly) // 12)
ax1.set_xticks(range(0, len(monthly), step))
ax1.set_xticklabels(monthly['year_month_str'][::step], rotation=45, fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Chart 2: Revenue by Region ---
ax2 = fig.add_subplot(gs[1, 0])
region_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
bars = ax2.bar(region_rev.index, region_rev.values / 1e6, color=sns.color_palette('Set2', 4))
ax2.set_title('Revenue by Region', fontsize=12, fontweight='bold')
ax2.set_ylabel('Revenue (₹M)')
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'₹{bar.get_height():.0f}M', ha='center', fontsize=9)

# --- Chart 3: Revenue by Product ---
ax3 = fig.add_subplot(gs[1, 1])
product_rev = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
ax3.barh(product_rev.index, product_rev.values / 1e6, color=sns.color_palette('viridis', 5))
ax3.set_title('Revenue by Product', fontsize=12, fontweight='bold')
ax3.set_xlabel('Revenue (₹M)')

# --- Chart 4: Top 10 Sales Reps ---
ax4 = fig.add_subplot(gs[1, 2])
top_reps = df.groupby('sales_rep')['revenue'].sum().sort_values(ascending=False).head(10)
ax4.barh(top_reps.index, top_reps.values / 1e6, color='coral')
ax4.set_title('Top 10 Sales Reps', fontsize=12, fontweight='bold')
ax4.set_xlabel('Revenue (₹M)')
ax4.invert_yaxis()

# --- Chart 5: Yearly Revenue Growth ---
ax5 = fig.add_subplot(gs[2, 0])
yearly = df.groupby('year')['revenue'].sum()
growth = yearly.pct_change() * 100
ax5.bar(yearly.index, yearly.values / 1e6, color=['#2196F3', '#4CAF50', '#FF9800'])
ax5.set_title('Yearly Revenue', fontsize=12, fontweight='bold')
ax5.set_ylabel('Revenue (₹M)')
for i, (yr, val) in enumerate(yearly.items()):
    ax5.text(yr, val / 1e6 + 1, f'₹{val/1e6:.0f}M', ha='center', fontsize=9)

# --- Chart 6: Monthly Avg by Product ---
ax6 = fig.add_subplot(gs[2, 1])
pivot = df.pivot_table(values='revenue', index='month', columns='product', aggfunc='sum') / 1e6
pivot.plot(ax=ax6, linewidth=1.5)
ax6.set_title('Monthly Revenue by Product', fontsize=12, fontweight='bold')
ax6.set_xlabel('Month')
ax6.set_ylabel('Revenue (₹M)')
ax6.legend(fontsize=7)

# --- Chart 7: Revenue Share Pie ---
ax7 = fig.add_subplot(gs[2, 2])
region_share = df.groupby('region')['revenue'].sum()
ax7.pie(region_share.values, labels=region_share.index, autopct='%1.1f%%',
        colors=sns.color_palette('pastel', 4), startangle=90)
ax7.set_title('Region Revenue Share', fontsize=12, fontweight='bold')

plt.savefig('sales_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 4: KPI Summary
# ============================================================

print("\n" + "="*50)
print("  KEY PERFORMANCE INDICATORS")
print("="*50)
print(f"  Total Revenue     : ₹{df['revenue'].sum()/1e6:.2f}M")
print(f"  Total Units Sold  : {df['units_sold'].sum():,}")
print(f"  Top Region        : {region_rev.idxmax()}")
print(f"  Top Product       : {product_rev.idxmax()}")
print(f"  Top Sales Rep     : {top_reps.idxmax()}")
print(f"  Avg Monthly Rev   : ₹{monthly['revenue'].mean()/1e6:.2f}M")
print("="*50)
