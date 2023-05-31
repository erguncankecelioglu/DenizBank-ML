from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd


def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath, delimiter=';')

    # Save customer_id data
    customer_id_data = df['uuid']

    # Drop unnecessary columns
    df = df.drop(['worst_status_active_inv', 'avg_payment_span_0_3m', 'account_incoming_debt_vs_paid_0_24m', 'name_in_email', 'uuid'], axis=1)

    # Convert 'has_paid' to binary
    df['has_paid'] = df['has_paid'].map({'TRUE': 1, 'FALSE': 0})

    # Remove rows with empty 'default' values
    nan_indices = df[df['default'].isna()].index
    df = df.drop(nan_indices)
    customer_id_data = customer_id_data.drop(nan_indices)

    # Replace NA values with 0 for specified columns
    columns_to_zero_fill = ["account_status", "account_worst_status_0_3m", "account_worst_status_12_24m",
                            "account_worst_status_3_6m","account_worst_status_6_12m"]
    df[columns_to_zero_fill] = df[columns_to_zero_fill].fillna(0)

    # Replace NA values with column average for specified columns
    columns_to_average_fill = ["avg_payment_span_0_12m", "num_arch_written_off_12_24m",
                               "num_arch_written_off_0_12m", "num_active_div_by_paid_inv_0_12m",
                               "account_days_in_term_12_24m", "account_days_in_rem_12_24m"]
    for column in columns_to_average_fill:
        df[column] = df[column].fillna(df[column].mean())

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Transform data
    y = df['default']
    X = df.drop('default', axis=1)

    # Define preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = ['merchant_category', 'merchant_group']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X = preprocessor.fit_transform(X)

    # Saving customer_id and processed data frame into csv file
    customer_id_data.to_csv('customer_id_data.csv', index=False)
    processed_df = pd.DataFrame(X)
    processed_df.to_csv('processed_dataset.csv', index=False)

    return X, y
