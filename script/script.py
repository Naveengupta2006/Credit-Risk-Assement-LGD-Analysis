# import all mandatory libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# load and concat data
main = pd.read_csv('../data/main_loan_base.csv')
main

monthly = pd.read_csv('../data/monthly_balance_base.csv')
monthly

repayment = pd.read_csv('../data/repayment_base.csv')
repayment

df = pd.concat([main, monthly, repayment], axis= 1)
df

# rename the column
df.rename(columns={'loan_acc_num':'account_name'}, inplace= True)

# Data Cleaning

# Remove duplicates
df.drop_duplicates(subset= 'account_name')

# handle missing value 
df['default_date'].isnull().sum()

df.dropna(subset=['default_date'], inplace= True)
df

# convert disbursal_date, repayment_date into datetime
df['disbursal_date'] = pd.to_datetime(df['disbursal_date'])
df['default_date'] = pd.to_datetime(df['default_date'])
df

# Outliers detection loan amount

# 1. calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['loan_amount'].quantile(0.25)
Q3 = df['loan_amount'].quantile(0.75)

# 2. Calculate the interquartile range(IQR)
IQR = Q3 - Q1

# 3. define the lower and upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 - 1.5 * IQR

# 4. filter and display the outliers
outliers = df[(df['loan_amount'] < lower_bound) | (df['loan_amount'] > upper_bound)]

print(f"lower bound: {lower_bound}")
print(f"upper bound: {upper_bound}")
print(f"Total Outliers found: {len(outliers)}")


# comparison plot 
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1.  boxplot to detect outliers points
sns.boxplot(x= df['loan_amount'], ax=axes[0], color='skyblue')
axes[0].set_title('Boxplot: spotting the whales')

# 2. histogram shape and skew
sns.histplot(df['loan_amount'], kde=True, ax=axes[1], color='orange')
axes[1].set_title('Histogram: vaisualizing the skew')

# apply the log tranformation on loan_amount column.
df['loan_amount_log'] = np.log1p(df['loan_amount'])

# visiualization
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='loan_amount_log', 
            hue='loan_type', fill= True, 
            common_norm=False,
            alpha=0.5)

plt.title('log of loan amount grouped by category')
plt.xlabel('log loan amount')
plt.ylabel('density')
plt.show()

# outliers detection for collateral_vale

# 1. calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['collateral_value'].quantile(0.25)
Q3 = df['collateral_value'].quantile(0.75)

# 2. Calculate the interquartile range(IQR)
IQR = Q3 - Q1

# 3. define the lower and upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 - 1.5 * IQR

# 4. filter and display the outliers
outliers = df[(df['collateral_value'] < lower_bound) | (df['collateral_value'] > upper_bound)]

print(f"lower bound: {lower_bound}")
print(f"upper bound: {upper_bound}")
print(f"Total Outliers found: {len(outliers)}")


# comparison plot 
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1.  boxplot to detect outliers points
sns.boxplot(x= df['collateral_value'], ax=axes[0], color='green')
axes[0].set_title('Boxplot: spotting the whales')

# 2. histogram shape and skew
sns.histplot(df['collateral_value'], kde=True, ax=axes[1], color='red')
axes[1].set_title('Histogram: vaisualizing the skew')

# apply the log transformation.
# apply the log tranformation on loan_amount column.
df['collateral_value_log'] = np.log1p(df['collateral_value'])

# visiualization
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='collateral_value_log', 
            hue='loan_type', fill= True, 
            common_norm=False,
            alpha=0.5)

plt.title('log of loan amount grouped by category')
plt.xlabel('log loan amount')
plt.ylabel('density')
plt.show()

# feature engineering

# loan to collateral Ratio = loan_amount / collateral_value
df['loan_to_collateral_ratio'] = np.where(
    df['collateral_value'] > 0,
    df['loan_amount'] / df['collateral_value'],
    np.nan
)
df

# EMI Burden Indicator
df['expected_emi'] = df['loan_amount'] / (df['tenure_years'] * 12)
df

df['emi_burden_flag'] = np.where(
    (df['monthly_emi'] > df['expected_emi']) &
    (df['missed_repayments'] > 0),1,0
)
df

# bounce risk score = based on cheques_bounces
# 0      0 = low risk
# 1-2    1 = medium risk
# >=3    2 = high risk
df['bounce_risk_score'] = np.select(
    [
        df['cheque_bounces'] == 0,
        df['cheque_bounces'].between(1, 2),
        df['cheque_bounces'] >= 3
    ],
    [0,1,2],
    default=0

)
df

# repayment stress score = missed_repayments
'''
missed repayment              stress score
0                              0 (low stress)
1-2                            1 (medium stress)
3-5                            2 (high stress)
>5                             3 (severe stress)
'''

df['repayment_stress_score'] = np.select(
    [
        df['missed_repayments'] == 0,
        df['missed_repayments'].between(1, 2),
        df['missed_repayments'].between(3, 5),
        df['missed_repayments'] > 5
    ],
    [0, 1, 2, 3],
    default=0
)
df

# Loan Exposure = total loans per customer

exposure_df = df.groupby('customer_name').agg(
    total_exposure_amount = ('loan_amount','sum'),
    total_number_of_loans = ('loan_amount','count')
).reset_index()

print(exposure_df)

df = df.merge(exposure_df, on='customer_name', how='left')
df

# LGD calculation

# calculate recovery rate
df['recovery_rate'] = df['repayment_amount'] / df['loan_amount']
df

# calculate LGD
df['LGD'] = 1 - df['recovery_rate']
df

# segement customers
def segment_lgd(lgd_value):
    if lgd_value > 0.6:
        return 'High'
    elif lgd_value > 0.3:
        return 'Medium'
    else:
        return 'low'
df['lgd_value'] = df['LGD'].apply(segment_lgd) # apply the 'high', 'medium', 'low'.
df    

# Risk segmentation

def perform_risk_segmentation(customer_data):

    df = customer_data.copy()

    # calculate collateral coverage (collateral value / total loan amount)
    df['collateral_coverage'] = df['collateral_value'] / (df['loan_amount'] + 0.001) # added 0.001 to avoid division by zero error

    # these threshold numbers are examples.
    condition_high_collateral = df['collateral_coverage']>= 1.5
    condition_high_bounces = df['cheque_bounces'] >= 3
    condition_high_missed = df['missed_repayments'] >= 4
    condition_multiple_loans = df['number_of_loans'] >= 3

    # apply the segmentation tags using numpy select function
    conditions = [
        condition_high_bounces | condition_high_missed, # high risk priority
        condition_multiple_loans, # medium-high risk
        condition_high_collateral # low risk
    ]

    choices = [
        'High Risk (Bounces/Missed Payments)',
        'Watch list (Multiple loans)',
        'Low Risk (High Collteral)'
    ]

    df['risk_segment'] = np.select(conditions, choices, default= 'Standard Risk')
    return df
perform_risk_segmentation(df)

# Correlation Analysis

def perform_correlation_analysis(df):
    """
    calculate and print the correlation between specific risk factores and outcome
    (LGD and Default)
    """
    print('--- Correlation Anaysis Result ---')

    # loan amount vs LGD
    corr_loan_lgd = df['loan_amount'].corr(df['LGD'])
    print(f"Loan amount VS LGD Correlations: {corr_loan_lgd}")

    # collateral value vs LGD
    corr_collateral_lgd = df['collateral_value'].corr(df['LGD'])
    print(f"Collateral value vs LGD correlations:{corr_collateral_lgd}")

    # missed repayment vs deafult
    corr_missed_lgd = df['missed_repayments'].corr(df['default_date'])
    print(f"Missed Repayment vs default correlation: {corr_missed_lgd:.2f}")

    # cheque bounces vs default
    corr_cheque_deafault = df['cheque_bounces'].corr(df['default_date'])
    print(f"Cheque Bounces vs Default Correlations: {corr_cheque_deafault}")

    print('-'*35)

perform_correlation_analysis(df)

# statistical & probability analysis
def descriptive_sta(df):

    # mean loan amount
    mean_loan = df['loan_amount'].mean()
    print(f"Mean loan amount: {mean_loan}")

    # median collateral_value
    median_collateral = df['collateral_value'].median()
    print(f"Median collateral value: {median_collateral}")

    # standard deviation of EMI
    std_emi = df['monthly_emi'].std()
    print(f"Standard Deviation of EMI: {std_emi:.2f}")

    # Distribution of missed repayment
    distribution_missed = df['repayment_amount'].value_counts().sort_index()
    print("\n Distribution of missed repayment (Number of missed : count of people):")
    print(distribution_missed)

descriptive_sta(df)    

def probability_analysis(df):

    # calculate collateral ratio
    df['collateral_ratio'] = df['collateral_value'] / df['loan_amount']
    df['default'] = df['default_date'].notna().astype(int)

    print("---Data with collateral ration---")
    print(df[['loan_amount','collateral_value', 'collateral_ratio', 'cheque_bounces','default']],  "\n")

    #calculate probabilities

    # probability of default
    p_default = df['default'].mean()
    print(f"Probability of default: {p_default:.2%}")

    # P(default | high cheque bounces)
    high_bounces_df = df[df['cheque_bounces'] >= 2]
    p_default_given_high_bounces = high_bounces_df['default'].mean()
    print(f"P(Default | High cheque bounces): {p_default_given_high_bounces:.2%}")

    # P(default | low collteral ratio)
    low_collateral_df = df[df['collateral_ratio'] < 1.1]
    p_default_given_low_collateral = low_collateral_df['default'].mean()
    print(f"P(default | low collateral ratio): {p_default_given_low_collateral:.2%}")

probability_analysis(df)

# Hypothesis Testing

# H0: collateral value has no impact on LGD
# H1: collateral value significantly affects LGD

'''
-- correlation significance
-- T-test between defaulters & non-defaulters
'''

print("--- 1. Correlation Significance ---")
# Test: Does collateral_value significantly affect LGD?
df = df[df['default'] == 1]

correlation, p_value= st.pearsonr(df['collateral_value'], df['LGD'])

print(f"Corrlation coefficient: {correlation}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Result: Reject H0. collateral value significantly affect LGD.")
else:
    print("Result: Fail to reject H).Not enough evidence to ay it affect LGD.")
        

print("\n--- 2. T-test (defaulters vs Non-defaulters) ---")

# sepearate our data into the two group 
defaulters_ratio = df[df['default'] == 1]['collateral_value']
non_defaulters_ratio = df[df['default'] == 0]['collateral_value']

# run the t-test
t_stat, p_value_ttest = st.ttest_ind(defaulters_ratio, non_defaulters_ratio)

print(f"T-statistics: {t_stat}")
print(f"P-value: {p_value_ttest}")

if p_value_ttest < 0.05:
    print("Result: There is a significant difference between the two groups.")
else:
    print("Result: No significant difference found between the groups.")

