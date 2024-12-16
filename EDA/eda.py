import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("eda_dataset.csv")

df['year'] = df['case_number'].str[:4].astype(int)

print(df.head(5))

percentage = df['has_facts_header'].value_counts(normalize=True)*100
percentage_by_year = df.groupby('year')['has_facts_header'].value_counts(normalize=True).unstack()*100

print(percentage_by_year)

percentage_by_year.plot(kind='bar', stacked=True, color=['orange', 'blue'])

# Add labels and title
plt.ylabel('Percentage')
plt.title('Percentage of True/False Values per Year')
plt.legend(title='has_facts_header', loc='upper left')

# Show the plot
plt.show()