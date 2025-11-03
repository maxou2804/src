import pandas as pd
import os 
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime

ratio_collection=[]
radius_collection=[]
rate_collection=[]
directory='C:\\Users\\trique\\Downloads\\MASTER_THESIS\\data_vizualization\\src_EDEN\\outputs_clusters'

data=pd.DataFrame({'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]})

data = data.sort_values('City')


for filename in os.listdir(directory):

    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print(filename)
        pandas_df=pd.read_csv(filepath)
        df=pd.DataFrame(pandas_df)
        area_init=df.loc[df['year'] == 1985,'area_km2']
        area_finit=df.loc[ df['year'] == 2014, 'area_km2']
        
        rate=(area_finit[290]-area_init[0])/area_init[0]
        radius=df.loc[df['year'] == 1985,'radial_distance_km']
        rate_collection.append(rate)
        
        ratio_area=area_init[0]/area_init[1:].sum()
        
        distance_avg=radius[1:].sum()/((len(radius)-1)*area_finit[290]**0.5)
        ratio_collection.append(ratio_area)
        radius_collection.append(distance_avg)
        
 


data['ratio_1985']=ratio_collection
data['radius_1985']=radius_collection
data['LCC_growth _rate']=rate_collection 



# fit_ratio=np.polyfit(data['ratio_1985'],data['beta'],1)


# plt.plot(data['ratio_1985'],data['beta'],'o')
# plt.plot(data['ratio_1985'],fit_ratio[0]*data['ratio_1985']+fit_ratio[1],'-',label=f'fit: y={fit_ratio[0]:.2f}x + {fit_ratio[1]:.2f}')
# plt.xlabel('ratio area 1985')
# plt.ylabel('beta exponent')
# plt.legend()
# plt.show()

# fit=np.polyfit(data['radius_1985'],data['beta'],1)

# plt.plot(data['radius_1985'],data['beta'],'o')
# plt.plot(data['radius_1985'],fit[0]*data['radius_1985']+fit[1],'-',label=f'fit: y={fit[0]:.2f}x + {fit[1]:.2f}')
# plt.xlabel('radius_1985')
# plt.ylabel('beta exponent')
# plt.legend()
# plt.show()


# plt.plot(data['radius_1985'],data['ratio_1985'],'o')
# plt.xlabel('radius_1985')   
# plt.ylabel('ratio area 1985')
# plt.show()

print("--- Correlation Analysis for initial state 1985 ---")
print("PEARSON CORRELATION: ")

corr_radius=np.corrcoef(data['radius_1985'],data['beta'])
corr_ratio=np.corrcoef(data['ratio_1985'],data['beta'])
corr_bonus=np.corrcoef(data['radius_1985'],data['ratio_1985'])

       
print(f'correlation radius vs beta: {corr_radius[0,1]}')
print(f'correlation ratio vs beta: {corr_ratio[0,1]}')
print(f'correlation radius vs ratio: {corr_bonus[0,1]}')

print()

print("SPEARMAN CORRELATION:")
res_radius=stats.spearmanr(data['radius_1985'],data['beta'])
res_ratio= stats.spearmanr(data['ratio_1985'],data['beta'])  
res_nonus= stats.spearmanr(data['radius_1985'],data['ratio_1985'])

print(f'correlation radius vs beta: {res_radius.correlation}')
print(f'correlation ratio vs beta: {res_ratio.correlation}')    
print(f'correlation radius vs ratio: {res_nonus.correlation}')

print()

print(f'p value radius vs beta: {res_radius.pvalue}')
print(f'p value ratio vs beta: {res_ratio.pvalue}') 
print(f'p value radius vs ratio: {res_nonus.pvalue}')
     

fit_rate=np.polyfit(data['LCC_growth _rate'],data['beta'],1)


# plt.plot(data['LCC_growth _rate'],data['beta'],'o')
# plt.plot(data['LCC_growth _rate'],fit_rate[0]*data['LCC_growth _rate']+fit_rate[1],'-',label=f'fit: y={fit_rate[0]:.2f}x + {fit_rate[1]:.2f}')
# plt.xlabel('LCC growth rate (1985-2015)')  
# plt.ylabel('beta exponent')
# plt.legend()
# plt.show()

print()
print("--- Correlation Analysis for LCC growth rate ---")
print("PEARSON CORRELATION: ")
corr_rate=np.corrcoef(data['LCC_growth _rate'],data['beta'])
print(f'correlation LCC growth rate vs beta: {corr_rate[0,1]}')
print()
print("SPEARMAN CORRELATION:")
res_rate= stats.spearmanr(data['LCC_growth _rate'],data['beta'])
print(f'correlation LCC growth rate vs beta: {res_rate.correlation}')
print(f'p value LCC growth rate vs beta: {res_rate.pvalue}')


# ============================================================================
# GENERATE OUTPUT FILES
# ============================================================================

def interpret_correlation(rho, p_value):
    """Interpret correlation strength and significance"""
    # Strength interpretation
    abs_rho = abs(rho)
    if abs_rho < 0.20:
        strength = "Very weak"
    elif abs_rho < 0.40:
        strength = "Weak"
    elif abs_rho < 0.60:
        strength = "Moderate"
    elif abs_rho < 0.80:
        strength = "Strong"
    else:
        strength = "Very strong"
    
    # Direction
    direction = "positive" if rho > 0 else "negative"
    
    # Significance
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p ≥ 0.05)"
    
    return strength, direction, significance


# Create formatted text report
output_text = []
output_text.append("="*80)
output_text.append("CORRELATION ANALYSIS REPORT")
output_text.append("Urban Morphology and Growth Patterns")
output_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_text.append("="*80)
output_text.append("")
output_text.append(f"Sample size: {len(data)} cities")
output_text.append("")

# Section 1: Initial State Analysis (1985)
output_text.append("="*80)
output_text.append("SECTION 1: INITIAL STATE ANALYSIS (1985)")
output_text.append("="*80)
output_text.append("")

# Radius vs Beta
output_text.append("-" * 80)
output_text.append("1.1 RADIUS vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_radius[0,1]:.4f}")
output_text.append(f"  Spearman correlation:  rho = {res_radius.correlation:.4f}")
output_text.append(f"  P-value:               p   = {res_radius.pvalue:.4f}")
strength, direction, significance = interpret_correlation(res_radius.correlation, res_radius.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Ratio vs Beta
output_text.append("-" * 80)
output_text.append("1.2 RATIO AREA vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_ratio[0,1]:.4f}")
output_text.append(f"  Spearman correlation:  rho = {res_ratio.correlation:.4f}")
output_text.append(f"  P-value:               p   = {res_ratio.pvalue:.4f}")
strength, direction, significance = interpret_correlation(res_ratio.correlation, res_ratio.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Radius vs Ratio
output_text.append("-" * 80)
output_text.append("1.3 RADIUS vs RATIO AREA")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_bonus[0,1]:.4f}")
output_text.append(f"  Spearman correlation:  rho = {res_nonus.correlation:.4f}")
output_text.append(f"  P-value:               p   = {res_nonus.pvalue:.4f}")
strength, direction, significance = interpret_correlation(res_nonus.correlation, res_nonus.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Section 2: Growth Rate Analysis
output_text.append("="*80)
output_text.append("SECTION 2: LCC GROWTH RATE ANALYSIS (1985-2015)")
output_text.append("="*80)
output_text.append("")

output_text.append("-" * 80)
output_text.append("2.1 LCC GROWTH RATE vs BETA EXPONENT")
output_text.append("-" * 80)
output_text.append(f"  Pearson correlation:   r   = {corr_rate[0,1]:.4f}")
output_text.append(f"  Spearman correlation:  rho = {res_rate.correlation:.4f}")
output_text.append(f"  P-value:               p   = {res_rate.pvalue:.4f}")
strength, direction, significance = interpret_correlation(res_rate.correlation, res_rate.pvalue)
output_text.append(f"  Interpretation:        {strength} {direction} correlation, {significance}")
output_text.append("")

# Summary section
output_text.append("="*80)
output_text.append("SUMMARY OF FINDINGS")
output_text.append("="*80)
output_text.append("")
output_text.append("Correlation Strength Guidelines (Spearman's ρ):")
output_text.append("  |ρ| < 0.20    Very weak")
output_text.append("  0.20-0.39     Weak")
output_text.append("  0.40-0.59     Moderate")
output_text.append("  0.60-0.79     Strong")
output_text.append("  0.80-1.00     Very strong")
output_text.append("")
output_text.append("Statistical Significance:")
output_text.append("  p < 0.05      Statistically significant")
output_text.append("  p < 0.01      Very significant")
output_text.append("  p < 0.001     Highly significant")
output_text.append("")
output_text.append("="*80)

# Write to text file
report_text = "\n".join(output_text)
with open('correlation_analysis_report_clusters.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*80)
print("OUTPUT FILES GENERATED:")
print("="*80)
print("✓ correlation_analysis_report_clusters.txt - Detailed formatted report")

# Create CSV summary
results_summary = {
    'Variable_Pair': [
        'Radius_1985 vs Beta',
        'Ratio_1985 vs Beta',
        'Radius_1985 vs Ratio_1985',
        'LCC_Growth_Rate vs Beta'
    ],
    'Pearson_r': [
        corr_radius[0,1],
        corr_ratio[0,1],
        corr_bonus[0,1],
        corr_rate[0,1]
    ],
    'Spearman_rho': [
        res_radius.correlation,
        res_ratio.correlation,
        res_nonus.correlation,
        res_rate.correlation
    ],
    'P_value': [
        res_radius.pvalue,
        res_ratio.pvalue,
        res_nonus.pvalue,
        res_rate.pvalue
    ],
    'Significant_at_0.05': [
        'Yes' if res_radius.pvalue < 0.05 else 'No',
        'Yes' if res_ratio.pvalue < 0.05 else 'No',
        'Yes' if res_nonus.pvalue < 0.05 else 'No',
        'Yes' if res_rate.pvalue < 0.05 else 'No'
    ]
}







results_df = pd.DataFrame(results_summary)
results_df.to_csv('correlation_results_summary.csv', index=False)
print("✓ correlation_results_summary.csv - Tabular summary for further analysis")
print("="*80)

# Also save the full data with all computed variables
data.to_csv('city_data_complete.csv', index=False)
print("✓ city_data_complete.csv - Complete dataset with all variables")
print("="*80)