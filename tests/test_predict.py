import application
import numpy as np
import pandas as pd

def test_prepare_features():

    data = {
        'Place_ID X Date': {0: '010Q650 X 2020-01-02'},
        'Date': {0: '2020-01-02'},
        'Place_ID': {0: '010Q650'},
        'precipitable_water_entire_atmosphere': {0: 11.0},
        'relative_humidity_2m_above_ground': {0: 60.20000076293945},
        'specific_humidity_2m_above_ground': {0: 0.0080399997532367},
        'temperature_2m_above_ground': {0: 18.5168395996094},
        'u_component_of_wind_10m_above_ground': {0: 1.996376872062683},
        'v_component_of_wind_10m_above_ground': {0: -1.2273949384689329},
        'L3_NO2_NO2_column_number_density': {0: 7.383037530673221e-05},
        'L3_NO2_NO2_slant_column_number_density': {0: 0.0001558203070035},
        'L3_NO2_absorbing_aerosol_index': {0: -1.2313302651595208},
        'L3_NO2_cloud_fraction': {0: 0.0065067959126918},
        'L3_NO2_sensor_altitude': {0: 840209.8746190539},
        'L3_NO2_sensor_azimuth_angle': {0: 76.53751201320065},
        'L3_NO2_sensor_zenith_angle': {0: 38.634284181059655},
        'L3_NO2_solar_azimuth_angle': {0: -61.73671867924881},
        'L3_NO2_solar_zenith_angle': {0: 22.358167327983335},
        'L3_NO2_stratospheric_NO2_column_number_density': {0: 5.679268210852429e-05},
        'L3_NO2_tropopause_pressure': {0: 6156.07421875},
        'L3_NO2_tropospheric_NO2_column_number_density': {0: 1.7037692852823708e-05},
        'L3_O3_O3_column_number_density': {0: 0.119094866975812},
        'L3_O3_O3_effective_temperature': {0: 234.15110215578883},
        'L3_O3_cloud_fraction': {0: 0.0},
        'L3_O3_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_O3_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_O3_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_O3_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CO_CO_column_number_density': {0: 0.0210802548217309},
        'L3_CO_H2O_column_number_density': {0: 883.332451347555},
        'L3_CO_cloud_height': {0: 267.01718371313405},
        'L3_CO_sensor_altitude': {0: 840138.4610518441},
        'L3_CO_sensor_azimuth_angle': {0: 74.54339259764188},
        'L3_CO_sensor_zenith_angle': {0: 38.62245099529423},
        'L3_CO_solar_azimuth_angle': {0: -61.789015833803646},
        'L3_CO_solar_zenith_angle': {0: 22.379054497267006},
        'L3_HCHO_HCHO_slant_column_number_density': {0: -1.041264324166104e-05},
        'L3_HCHO_cloud_fraction': {0: 0.0},
        'L3_HCHO_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_HCHO_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_HCHO_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_HCHO_solar_zenith_angle': {0: 22.36366476694216},
        'L3_HCHO_tropospheric_HCHO_column_number_density': {0: 6.38880016719335e-05},
        'L3_HCHO_tropospheric_HCHO_column_number_density_amf': {0: 0.5668279494200223},
        'L3_CLOUD_cloud_base_height': {0: 38.0},
        'L3_CLOUD_cloud_base_pressure': {0: 38.0},
        'L3_CLOUD_cloud_fraction': {0: 0.0},
        'L3_CLOUD_cloud_optical_depth': {0: 38.0},
        'L3_CLOUD_cloud_top_height': {0: 38.0},
        'L3_CLOUD_cloud_top_pressure': {0: 38.0},
        'L3_CLOUD_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_CLOUD_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_CLOUD_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_CLOUD_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CLOUD_surface_albedo': {0: 38.0},
        'L3_AER_AI_absorbing_aerosol_index': {0: -1.231329983630794},
        'L3_AER_AI_sensor_altitude': {0: 840209.8746190539},
        'L3_AER_AI_sensor_azimuth_angle': {0: 76.53751201320065},
        'L3_AER_AI_sensor_zenith_angle': {0: 38.634284181059655},
        'L3_AER_AI_solar_azimuth_angle': {0: -61.73671867924881},
        'L3_AER_AI_solar_zenith_angle': {0: 22.358167327983335},
        'L3_SO2_SO2_column_number_density': {0: -0.0001268544881504},
        'L3_SO2_SO2_column_number_density_amf': {0: 0.3125208453527509},
        'L3_SO2_SO2_slant_column_number_density': {0: -4.046582245013976e-05},
        'L3_SO2_absorbing_aerosol_index': {0: -1.861475654115453},
        'L3_SO2_cloud_fraction': {0: 0.0},
        'L3_SO2_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_SO2_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_SO2_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_SO2_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CH4_CH4_column_volume_mixing_ratio_dry_air': {0: 1793.7935791015625},
        'L3_CH4_aerosol_height': {0: 3227.85546875},
        'L3_CH4_aerosol_optical_depth': {0: 0.0105790393427014},
        'L3_CH4_sensor_azimuth_angle': {0: 74.48104858398438},
        'L3_CH4_sensor_zenith_angle': {0: 37.50149917602539},
        'L3_CH4_solar_azimuth_angle': {0: -62.14263916015625},
        'L3_CH4_solar_zenith_angle': {0: 22.54511833190918}
    }

    df = pd.DataFrame(data)

    df['Date'] = pd.to_datetime(df['Date'])

    df['saleyear'] = df.Date.dt.year
    df['salemonth'] = df.Date.dt.month
    df['saleday'] = df.Date.dt.day
    df['saledayofweek'] = df.Date.dt.day_of_week
    df['saledayofyear'] = df.Date.dt.day_of_year

    df= df.drop(columns=['L3_SO2_SO2_column_number_density_amf','L3_NO2_stratospheric_NO2_column_number_density','L3_CLOUD_sensor_azimuth_angle','L3_HCHO_solar_azimuth_angle','L3_SO2_sensor_zenith_angle','L3_CH4_solar_zenith_angle','temperature_2m_above_ground','L3_NO2_absorbing_aerosol_index','L3_CO_sensor_azimuth_angle','L3_AER_AI_sensor_zenith_angle','L3_AER_AI_solar_azimuth_angle','L3_AER_AI_solar_zenith_angle','L3_CO_sensor_zenith_angle','L3_HCHO_sensor_zenith_angle','L3_HCHO_tropospheric_HCHO_column_number_density_amf','L3_SO2_sensor_azimuth_angle','L3_SO2_absorbing_aerosol_index','L3_CH4_sensor_zenith_angle','specific_humidity_2m_above_ground','L3_NO2_NO2_slant_column_number_density','L3_NO2_cloud_fraction','L3_NO2_tropopause_pressure','L3_O3_sensor_azimuth_angle','L3_O3_cloud_fraction','L3_CLOUD_sensor_zenith_angle','L3_CLOUD_solar_azimuth_angle','L3_CLOUD_solar_zenith_angle','L3_CO_sensor_altitude','L3_HCHO_sensor_azimuth_angle','L3_CO_solar_zenith_angle','L3_HCHO_cloud_fraction','L3_HCHO_solar_zenith_angle','L3_HCHO_tropospheric_HCHO_column_number_density','L3_SO2_SO2_column_number_density','L3_CLOUD_cloud_top_height','L3_CLOUD_cloud_top_pressure','L3_AER_AI_absorbing_aerosol_index','L3_AER_AI_sensor_azimuth_angle','L3_SO2_SO2_slant_column_number_density','L3_SO2_cloud_fraction','L3_SO2_solar_zenith_angle','L3_CH4_aerosol_height','L3_CH4_sensor_azimuth_angle','L3_CH4_solar_azimuth_angle','Place_ID X Date','Date'], axis=1)
    
    cat = ['Place_ID','saleday', 'saledayofweek', 'saledayofyear']
    df[cat] = df[cat].astype(str)
    
    train_numerical_features=df.select_dtypes(include=['float']).columns
    train_numerical = []
    for num in train_numerical_features:
        train_numerical.append(num)

    train_categorical_features=df.select_dtypes(include=['object']).columns
    train_categorical= []
    for cat in train_categorical_features:
        train_categorical.append(cat)

    expected_features = df[train_categorical + train_numerical].to_dict(orient='records')

    actual_features = application.prepare_features(data)

    assert actual_features == expected_features

def test_predict():
    data = {
        'Place_ID X Date': {0: '010Q650 X 2020-01-02'},
        'Date': {0: '2020-01-02'},
        'Place_ID': {0: '010Q650'},
        'precipitable_water_entire_atmosphere': {0: 11.0},
        'relative_humidity_2m_above_ground': {0: 60.20000076293945},
        'specific_humidity_2m_above_ground': {0: 0.0080399997532367},
        'temperature_2m_above_ground': {0: 18.5168395996094},
        'u_component_of_wind_10m_above_ground': {0: 1.996376872062683},
        'v_component_of_wind_10m_above_ground': {0: -1.2273949384689329},
        'L3_NO2_NO2_column_number_density': {0: 7.383037530673221e-05},
        'L3_NO2_NO2_slant_column_number_density': {0: 0.0001558203070035},
        'L3_NO2_absorbing_aerosol_index': {0: -1.2313302651595208},
        'L3_NO2_cloud_fraction': {0: 0.0065067959126918},
        'L3_NO2_sensor_altitude': {0: 840209.8746190539},
        'L3_NO2_sensor_azimuth_angle': {0: 76.53751201320065},
        'L3_NO2_sensor_zenith_angle': {0: 38.634284181059655},
        'L3_NO2_solar_azimuth_angle': {0: -61.73671867924881},
        'L3_NO2_solar_zenith_angle': {0: 22.358167327983335},
        'L3_NO2_stratospheric_NO2_column_number_density': {0: 5.679268210852429e-05},
        'L3_NO2_tropopause_pressure': {0: 6156.07421875},
        'L3_NO2_tropospheric_NO2_column_number_density': {0: 1.7037692852823708e-05},
        'L3_O3_O3_column_number_density': {0: 0.119094866975812},
        'L3_O3_O3_effective_temperature': {0: 234.15110215578883},
        'L3_O3_cloud_fraction': {0: 0.0},
        'L3_O3_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_O3_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_O3_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_O3_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CO_CO_column_number_density': {0: 0.0210802548217309},
        'L3_CO_H2O_column_number_density': {0: 883.332451347555},
        'L3_CO_cloud_height': {0: 267.01718371313405},
        'L3_CO_sensor_altitude': {0: 840138.4610518441},
        'L3_CO_sensor_azimuth_angle': {0: 74.54339259764188},
        'L3_CO_sensor_zenith_angle': {0: 38.62245099529423},
        'L3_CO_solar_azimuth_angle': {0: -61.789015833803646},
        'L3_CO_solar_zenith_angle': {0: 22.379054497267006},
        'L3_HCHO_HCHO_slant_column_number_density': {0: -1.041264324166104e-05},
        'L3_HCHO_cloud_fraction': {0: 0.0},
        'L3_HCHO_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_HCHO_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_HCHO_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_HCHO_solar_zenith_angle': {0: 22.36366476694216},
        'L3_HCHO_tropospheric_HCHO_column_number_density': {0: 6.38880016719335e-05},
        'L3_HCHO_tropospheric_HCHO_column_number_density_amf': {0: 0.5668279494200223},
        'L3_CLOUD_cloud_base_height': {0: 38.0},
        'L3_CLOUD_cloud_base_pressure': {0: 38.0},
        'L3_CLOUD_cloud_fraction': {0: 0.0},
        'L3_CLOUD_cloud_optical_depth': {0: 38.0},
        'L3_CLOUD_cloud_top_height': {0: 38.0},
        'L3_CLOUD_cloud_top_pressure': {0: 38.0},
        'L3_CLOUD_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_CLOUD_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_CLOUD_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_CLOUD_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CLOUD_surface_albedo': {0: 38.0},
        'L3_AER_AI_absorbing_aerosol_index': {0: -1.231329983630794},
        'L3_AER_AI_sensor_altitude': {0: 840209.8746190539},
        'L3_AER_AI_sensor_azimuth_angle': {0: 76.53751201320065},
        'L3_AER_AI_sensor_zenith_angle': {0: 38.634284181059655},
        'L3_AER_AI_solar_azimuth_angle': {0: -61.73671867924881},
        'L3_AER_AI_solar_zenith_angle': {0: 22.358167327983335},
        'L3_SO2_SO2_column_number_density': {0: -0.0001268544881504},
        'L3_SO2_SO2_column_number_density_amf': {0: 0.3125208453527509},
        'L3_SO2_SO2_slant_column_number_density': {0: -4.046582245013976e-05},
        'L3_SO2_absorbing_aerosol_index': {0: -1.861475654115453},
        'L3_SO2_cloud_fraction': {0: 0.0},
        'L3_SO2_sensor_azimuth_angle': {0: 76.53642577586722},
        'L3_SO2_sensor_zenith_angle': {0: 38.59301738304029},
        'L3_SO2_solar_azimuth_angle': {0: -61.75258652266882},
        'L3_SO2_solar_zenith_angle': {0: 22.36366476694216},
        'L3_CH4_CH4_column_volume_mixing_ratio_dry_air': {0: 1793.7935791015625},
        'L3_CH4_aerosol_height': {0: 3227.85546875},
        'L3_CH4_aerosol_optical_depth': {0: 0.0105790393427014},
        'L3_CH4_sensor_azimuth_angle': {0: 74.48104858398438},
        'L3_CH4_sensor_zenith_angle': {0: 37.50149917602539},
        'L3_CH4_solar_azimuth_angle': {0: -62.14263916015625},
        'L3_CH4_solar_zenith_angle': {0: 22.54511833190918}
    }

    features = application.prepare_features(data)

    actual_features = application.predict(features)
    actual_features = np.round(actual_features, 2)

    expected_features = 48.2

    assert actual_features == expected_features
