# %%
# Competition link
# https://www.kaggle.com/c/bank-card-cancellation/overview/evaluation

# %%
## Libraries ----
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

## Constants ----
FULL_EXECUTION_REPORT = False

# %%
## Load data ----
# Train
df_train_sys1_1 = pd.read_csv('../data/train_sys1_part1of2.csv', delimiter=',', decimal='.', parse_dates=['date_of_birth'])
df_train_sys1_2 = pd.read_csv('../data/train_sys1_part2of2.csv', delimiter=',', decimal='.', parse_dates=['date_of_birth'])
df_train_sys2 = pd.read_csv('../data/train_sys2.csv', delimiter=',', decimal='.')
df_train_sys3 = pd.read_csv('../data/train_sys3.csv', delimiter=',', decimal='.')

# Test
df_test_sys1 = pd.read_csv('../data/test_sys1.csv', delimiter=',', decimal='.', parse_dates=['date_of_birth'])
df_test_sys2 = pd.read_csv('../data/test_sys2.csv', delimiter=',', decimal='.')
df_test_sys3 = pd.read_csv('../data/test_sys3.csv', delimiter=',', decimal='.')

# %%
# Train
df_train_sys1 = pd.concat([df_train_sys1_1, df_train_sys1_2])
df_train = pd.merge(df_train_sys1, df_train_sys2, on='id')
df_train = pd.merge(df_train, df_train_sys3, on='id')
print('Train '+str(df_train.shape))

#Test
df_test = pd.merge(df_test_sys1, df_test_sys2, on='id')
df_test = pd.merge(df_test, df_test_sys3, on='id')
print('Test '+str(df_test.shape))

# Train (5657, 22)
# Test (3970, 21)

# %%
## Report
if FULL_EXECUTION_REPORT:
    ProfileReport(df_train, title="Exploratory Data Analysis: 'Raw'", minimal=True).to_file("../eda/train.html")

# %%
## Feature engineering
### Dates
import datetime

df_train['date_of_birth_year'] = df_train['date_of_birth'].dt.year
df_train['date_of_birth_age'] = df_train['date_of_birth'].apply(lambda x : (pd.datetime.now() - x)).astype('<m8[Y]')

df_test['date_of_birth_year'] = df_test['date_of_birth'].dt.year
df_test['date_of_birth_age'] = df_test['date_of_birth'].apply(lambda x : (pd.datetime.now() - x)).astype('<m8[Y]')

# %%
# Filter columns that not exist in `test`
columns_to_drop = ['id_sce', 'Country', 'date_of_birth']
df_train_filtered = df_train.drop(columns=columns_to_drop)
df_test_filtered = df_test.drop(columns=columns_to_drop)

target = 'cancellation'

# %%
## Helpers ----
def tag_missing(df, fill_columns):
    cols_missing = list()
    for col in fill_columns:
        if sum(df[col].isnull()) > 0:
            cols_missing.append(col)
            df[col+'_missing'] = df[col].isna()*1
            df[col].fillna(df[col].mean(), inplace=True)
        
    return df, cols_missing

def tag_missing_string(df, fill_columns):
    cols_ = list()
    cols_missing = list()
    for col in fill_columns:
        if sum(df[col].isnull()) > 0:
            cols_.append(col)
            cols_missing.append(col+'_missing')
            df[col+'_missing'] = df[col].isna()*1
        
    return df, cols_, cols_missing

### Fill missing values ----
df_to_missing = df_train_filtered.select_dtypes(include=np.number)
df_train_numeric_missing,  df_train_numeric_missing_columns = tag_missing(df_to_missing, df_to_missing.columns)

df_to_missing = df_test_filtered.select_dtypes(include=np.number)
df_test_numeric_missing,  df_test_numeric_missing_columns = tag_missing(df_to_missing, df_to_missing.columns)

### Fill missing string ----
df_to_missing = df_train_filtered.select_dtypes(exclude=np.number)
df_train_string_missing,  _, df_train_string_missing_columns = tag_missing_string(df_to_missing, df_to_missing.columns)

df_to_missing = df_test_filtered.select_dtypes(exclude=np.number)
df_test_string_missing,  _, df_test_string_missing_columns = tag_missing_string(df_to_missing, df_to_missing.columns)

### Get dummies ----
df_to_dummies = df_train_filtered.select_dtypes(exclude=np.number)
df_train_dummies = pd.get_dummies(df_to_dummies, columns = df_to_dummies.columns, drop_first=True)

df_to_dummies = df_test_filtered.select_dtypes(exclude=np.number)
df_test_dummies = pd.get_dummies(df_to_dummies, columns = df_to_dummies.columns, drop_first=True)

### Concatenate missing and dummies ----
df_train_missing_dummies = pd.concat([df_train_numeric_missing, df_train_string_missing[df_train_string_missing_columns], df_train_dummies], axis=1)
df_test_missing_dummies = pd.concat([df_test_numeric_missing, df_test_string_missing[df_test_string_missing_columns], df_test_dummies], axis=1)

### Flag for `revolving_balance` ----
df_train_missing_dummies['revolving_balance_zero'] = (df_train_missing_dummies['revolving_balance'] == 0)*1
df_test_missing_dummies['revolving_balance_zero'] = (df_test_missing_dummies['revolving_balance'] == 0)*1

# %%
## To CSV for Dataiku ----
df_train_missing_dummies.to_csv('../data_transformed/df_train_missing_dummies.csv', index=False)
df_test_missing_dummies.to_csv('../data_transformed/df_test_missing_dummies.csv', index=False)

# %%