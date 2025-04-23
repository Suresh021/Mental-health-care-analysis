import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:\Users\Suresh Kumar\Downloads\Mental_Health_Care_in_the_Last_4_Weeks.csv')

# Objective 1: Exploratory Data Analysis
print(df.head(19),"\n")# Displays the first 19 rows
print(df.info(),"\n")# Structure of the dataset
print(df.describe(),"\n")# Summary statistics
print(df.shape,"\n")
print(df.isnull().sum(),"\n")#finds number of missing values in the dataset. 

#Handling Missing values
print(df.dropna(),"\n")
print(df['Value'].fillna(df['Value'].median()),"\n")

#Aggeregation and grouping
grouped_values = df.groupby("Group")["Value"].sum().sort_values()
print(grouped_values,"\n")

#value counts and unique values 
print(df['Indicator'].value_counts(),"\n")
print(df['Indicator'].unique(),"\n")

# Display data types of each column
print("\nData Types of Each Column:\n", df.dtypes)

# Optional: View example mismatched entries (e.g., numerical stored as object)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\nUnique entries in object column '{col}':\n", df[col].unique()[:10])

# Check for negative values in numerical column 'Value'
if 'Value' in df.columns:
    print("\nRows with Negative Values in 'Value' Column:")
    print(df[df['Value'] < 0])
        
#removing the duplicate values
print(df['Group'].duplicated().sum(),"\n")
print(df['Group'].drop_duplicates(),"\n")
#Filtering and sorting the values
filtered_data = df[df['Group']== "By State"]
print(f"{filtered_data.shape[0]}")
print(df.sort_values('LowCI',ascending = False),"\n")
print(df[['LowCI','HighCI']].corr())

# Objective 2: Analysis
#Temporal Trend
# Convert Time Period column to datetime format
df['Time Period'] = pd.to_datetime(df['Time Period'])

# Group by Time Period and calculate mean mental health care usage
df_trend = df.groupby('Time Period')['Value'].mean()

# Plot the trend
plt.figure(figsize=(12,6))
df_trend.plot(marker='o', title='Average Mental Health Care Usage Over Time')
plt.xlabel('Time Period')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()

#Compare Trends Across Demographic Factors
top_subgroups = df.groupby('Subgroup')['Value'].mean().nlargest(10).index

# Filter the dataframe to only include those top subgroups
filtered_df = df[df['Subgroup'].isin(top_subgroups)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x='Time Period Label', y='Value', hue='Subgroup', marker='o', ci=None, legend=None)
plt.title("Mental Health Care Usage by Top 10 Demographic Subgroups Over Time")
plt.xlabel("Time Period")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# Analyze the Impact of the Indicator
plt.figure(figsize=(20, 6))
sns.lineplot(data=df, x='Time Period Label', y='Value', hue='Indicator', marker='o', ci=None)
plt.title("Mental Health Care Usage by Type of Service Over Time")
plt.xlabel("Time Period")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()


# Objective 3: Descriptive Statistical Analysis
print("Mean: ",df['HighCI'].mean())#Mean for HighCI.
print("Mode: ",df['HighCI'].mode())#Mode for HighCI.
print("Median: ",df['HighCI'].median())#Median for HighCI.
print("Standard deviation: ",df['HighCI'].std())#Standard deviation of HighCI.
print("minimum: ",df['HighCI'].min())#minimum value from HighCI column.
print("Maximum: ",df['HighCI'].max())#maximum value from HighCI column.
print("Range: ",df['HighCI'].max()-df['HighCI'].min())#Range in the HighCI column.
print("confidence Interval: ",df[['LowCI','HighCI']].describe(),"\n")#Confidence Interval between LowCI and HighCI.


# Objective 4: Visualization
#Bar plot for Mental Health Care Usage for top 10 states
latest_data = filtered_data.sort_values('Time Period Start Date').groupby('State').last()
top_10 = latest_data.sort_values(by='Value', ascending=False).head(10)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10.index, y=top_10['Value'], palette='coolwarm')
plt.xlabel("State")
plt.ylabel("Value")
plt.title("Top 10 States - Mental Health Care Usage for top 10 states")
plt.xticks(rotation=45)
plt.show()

#Histogram for showing 
#X-axis: Ranges of the Value (percentage), broken into 20 bins.
#Y-axis: Count of rows in the dataset that fall into each value range.
plt.figure(figsize=(8,5))
sns.histplot(df["Value"], bins=20, color='skyblue', edgecolor='black', kde=True)
plt.title("Mental Health Care Usage by Value")
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

#Pie chart for Mental Health Care Usage by Group.
plt.figure(figsize=(8, 5))
grouped_values.plot(kind="pie",color='skyblue')
plt.xlabel('Group')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title("Mental Health Care Usage by Group")
plt.show()

#Scatter plot for representing the relation between Time Period and value
plt.figure(figsize=(8, 5))
plt.scatter(df['Time Period'], df['Value'], color='red', alpha=1)
plt.title("Time Period vs value")
plt.xlabel('Time Period')
plt.ylabel('Value')
plt.show()

top_groups = df['Group'].value_counts().nlargest(5).index.tolist()

# Filter the dataframe for only the top 5 groups
filtered_df = df[df['Group'].isin(top_groups)]

# Plotting the box plot for top 5 Groups
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Value', data=filtered_df)
plt.title('Box Plot of Value for Top 5 Groups')
plt.xlabel('Group')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#Heat map: correlation matrix of numerical variables
numerical_df = df.select_dtypes(include=['number'])
if 'Suppression Flag' in numerical_df.columns:
    numerical_df = numerical_df.drop(columns=['Suppression Flag'])

# Compute correlation matrix
corr_matrix = numerical_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, linewidth=0.5, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Columns")
plt.tight_layout()
plt.show()

# Objective 5: Statistical Testing
#T-Test for comparing the value for males vs females
from scipy.stats import ttest_ind

sex_data = df[df["Group"] == "By Sex"]

# Get 'Value' data for Male and Female, dropping missing values
male_values = sex_data[sex_data["Subgroup"] == "Male"]["Value"].dropna()
female_values = sex_data[sex_data["Subgroup"] == "Female"]["Value"].dropna()

# Perform independent Two-sample t-test 
t_stat, p_value = ttest_ind(male_values, female_values, equal_var=False)


#Display results
print("\nT-statistic:", t_stat)
print("P-value:", p_value)

# Decision based on significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The mean values are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in mean values.")

from statsmodels.stats.weightstats import ztest


#Z-Test for z-test between the proportions of people who took prescription medication vs. received counseling
#Filter values for the two indicators
medication = df[df["Indicator"] == "Took Prescription Medication for Mental Health, Last 4 Weeks"]["Value"].dropna()
counseling = df[df["Indicator"] == "Received Counseling or Therapy, Last 4 Weeks"]["Value"].dropna()

# Perform z-test
z_stat, p_value = ztest(medication, counseling)

# Display results
print(f"\nZ-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between medication and counseling proportions.")
else:
    print("Fail to reject the null hypothesis: No significant difference between medication and counseling proportions.")

#Chi-squred Test for significant association between Subgroup and Indicator when Group == "By Sex"
import scipy.stats as stats

sex_indicator_data = df[df["Group"] == "By Sex"]

# Create contingency table between Subgroup and Indicator
contingency_table = pd.crosstab(sex_indicator_data["Subgroup"], sex_indicator_data["Indicator"])

# Perform Chi-squared test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Output
print(f"\nChi2 = {chi2:.2f}")
print(f"p-value: {p:.4f}")

# Decision based on significance level
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is a significant association between gender and type of mental health service used.")
else:
    print("Fail to reject the null hypothesis: No significant association between gender and type of mental health service used.")


#Other Objectives
#Comparing mental health care access and behavior across different states or regions to identify geographical disparities in mental health support during the last 4 week
# Filter for Group = "By State" (to analyze by geographical region)
state_data = df[df["Group"] == "By State"]

# Group by State (Subgroup) and Indicator, then compute average value
state_summary = state_data.groupby(["Subgroup", "Indicator"])["Value"].mean().unstack()

# Add a column for average usage across all indicators
state_summary["Average"] = state_summary.mean(axis=1)

# Sort by average to find top states
state_summary_sorted = state_summary.sort_values("Average", ascending=False)

# Display top 10 states
print("\nTop 10 states by average mental health care usage:\n")
print(state_summary_sorted.head(10))


#Geographical Aggregation
state_avg = df.groupby('State')['Value'].mean().sort_values(ascending=False)
state_avg.plot(kind='bar', figsize=(12,6),title="Average Mental Health Indicator by State")
plt.show()

