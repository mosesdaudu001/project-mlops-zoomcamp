import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def fill_dataframe(df):
    numerical_columns=df.select_dtypes(include=['float64']).columns
    for col in df.columns:
        if col in numerical_columns:
            # For automaticaly filling all columns with dtype(int or float) with mean
            median=df[col].median()
            df[col]=df[col].fillna(median)
    return df

def read_dataframe(filename: str, train=False):
    df = pd.read_csv(filename, parse_dates=["Date"])

    df = fill_dataframe(df)

    if train:
        df=df.drop(columns=['target_min', 'target_max', 'target_variance', 'target_count'], axis=1)

    df['saleyear'] = df.Date.dt.year
    df['salemonth'] = df.Date.dt.month
    df['saleday'] = df.Date.dt.day
    df['saledayofweek'] = df.Date.dt.day_of_week
    df['saledayofyear'] = df.Date.dt.day_of_year

    df= df.drop(columns=['L3_SO2_SO2_column_number_density_amf','L3_NO2_stratospheric_NO2_column_number_density','L3_CLOUD_sensor_azimuth_angle','L3_HCHO_solar_azimuth_angle','L3_SO2_sensor_zenith_angle','L3_CH4_solar_zenith_angle','temperature_2m_above_ground','L3_NO2_absorbing_aerosol_index','L3_CO_sensor_azimuth_angle','L3_AER_AI_sensor_zenith_angle','L3_AER_AI_solar_azimuth_angle','L3_AER_AI_solar_zenith_angle','L3_CO_sensor_zenith_angle','L3_HCHO_sensor_zenith_angle','L3_HCHO_tropospheric_HCHO_column_number_density_amf','L3_SO2_sensor_azimuth_angle','L3_SO2_absorbing_aerosol_index','L3_CH4_sensor_zenith_angle','specific_humidity_2m_above_ground','L3_NO2_NO2_slant_column_number_density','L3_NO2_cloud_fraction','L3_NO2_tropopause_pressure','L3_O3_sensor_azimuth_angle','L3_O3_cloud_fraction','L3_CLOUD_sensor_zenith_angle','L3_CLOUD_solar_azimuth_angle','L3_CLOUD_solar_zenith_angle','L3_CO_sensor_altitude','L3_HCHO_sensor_azimuth_angle','L3_CO_solar_zenith_angle','L3_HCHO_cloud_fraction','L3_HCHO_solar_zenith_angle','L3_HCHO_tropospheric_HCHO_column_number_density','L3_SO2_SO2_column_number_density','L3_CLOUD_cloud_top_height','L3_CLOUD_cloud_top_pressure','L3_AER_AI_absorbing_aerosol_index','L3_AER_AI_sensor_azimuth_angle','L3_SO2_SO2_slant_column_number_density','L3_SO2_cloud_fraction','L3_SO2_solar_zenith_angle','L3_CH4_aerosol_height','L3_CH4_sensor_azimuth_angle','L3_CH4_solar_azimuth_angle','Place_ID X Date','Date'], axis=1)
    
    if train:
        col = ['u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground', 'L3_NO2_sensor_azimuth_angle', 'L3_O3_sensor_zenith_angle', 'L3_O3_solar_azimuth_angle','L3_CO_H2O_column_number_density', 'L3_CO_cloud_height','L3_HCHO_HCHO_slant_column_number_density', 'L3_CLOUD_cloud_base_height','L3_CLOUD_cloud_fraction', 'L3_CLOUD_cloud_optical_depth', 'L3_CH4_CH4_column_volume_mixing_ratio_dry_air']
        for co in col:
            q_cutoff = df[co].quantile(0.95)
            mask = df[co] < q_cutoff

            df = df[mask]

    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):

    cat = ['Place_ID','saleday', 'saledayofweek', 'saledayofyear']
    df[cat] = df[cat].astype(str)
    
    df = df.drop(columns='target', axis=1)
    train_numerical_features=df.select_dtypes(include=['float']).columns
    train_numerical = []
    for num in train_numerical_features:
        train_numerical.append(num)

    train_categorical_features=df.select_dtypes(include=['object']).columns
    train_categorical= []
    for cat in train_categorical_features:
        train_categorical.append(cat)

    dicts = df[train_categorical + train_numerical].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the trip data is saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load parquet files
    df_train = read_dataframe(
        os.path.join(raw_data_path, "Train.csv"), train=True
    )

    # df_test = read_dataframe(
    #     os.path.join(raw_data_path, "Test.csv")
    # )

    # Extract the target
    target = 'target'
    Y = df_train[target].values

    dv = DictVectorizer()
    X, dv = preprocess(df_train, dv, fit_dv=True)

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3, random_state=2)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))


if __name__ == '__main__':
    run_data_prep()


