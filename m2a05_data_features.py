import pandas as pd

# Load up the dataset setting correct header labels:
col_names = ['education', 'age', 'capital-gain', 'race',
             'capital-loss', 'hours-per-week', 'sex', 'classification']
df = pd.read_csv('datasets/census.data', names=col_names)
# Check the data type of all columns:
for i in df.columns:
    print(i)
    print(df[i].dtypes)
    print(df[i].unique())
df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
print(df.dtypes)
# Check potential categorical features:
edu_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th',
             '10th', '11th', '12th', 'HS-grad', 'Some-college',
             'Bachelors', 'Masters', 'Doctorate']
df['education'] = df['education'].astype('category',
                                         ordered=True,
                                         categories=edu_order).cat.codes
class_order = ['<=50K', '>50K']
df['classification'] = \
    df['classification'].astype('category',
                                ordered=True,
                                categories=class_order).cat.codes
df = pd.get_dummies(df, columns=['sex', 'race'])
print(df)
