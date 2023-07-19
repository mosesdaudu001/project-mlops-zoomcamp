# project-mlops-zoomcamp
This is the capstone project for my mlops-zoomcamp course with datatalks

# Capstone Project (Mlops-Zoomcamp) - Urban Air Quality Prediction

![Architecture](./images/air_quality-01.jpg)

## Problem Statement

This is a capstone project associated with [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp), and it will be peer reviewed and scored.

Air pollution is a critical global issue affecting the health and well-being of millions of people. The World Health Organization (WHO) estimates that more than 90% of the world's population lives in areas with air quality levels exceeding their guidelines, leading to numerous health problems, including respiratory and cardiovascular diseases. Therefore, it is imperative to develop accurate and efficient methods to monitor and predict air quality in cities worldwide.

The objective of this machine learning project is to create a predictive model that leverages satellite data to estimate PM2.5 particulate matter concentration in the air every day for each city. PM2.5 refers to atmospheric particulate matter that have a diameter of less than 2.5 micrometers and is one of the most harmful air pollutants. PM2.5 is a common measure of air quality that normally requires ground-based sensors to measure.

The successful completion of this project will lead to a powerful tool for predicting air quality in cities worldwide, helping local governments and environmental agencies take proactive measures to address pollution and safeguard public health. Moreover, it can provide valuable insights into the spatial and temporal patterns of air pollution, aiding in the development of effective mitigation strategies and sustainable urban planning.

## Dataset

The data covers the last three months, spanning hundreds of cities across the globe.

The data comes from three main sources:

1. Ground-based air quality sensors. These measure the target variable (PM2.5 particle concentration). In addition to the target column (which is the daily mean concentration) there are also columns for minimum and maximum readings on that day, the variance of the readings and the total number (count) of sensor readings used to compute the target value. This data is only provided for the train set - you must predict the target variable for the test set.
2. The Global Forecast System (GFS) for weather data. Humidity, temperature and wind speed, which can be used as inputs for your model.
3. The Sentinel 5P satellite. This satellite monitors various pollutants in the atmosphere. For each pollutant, we queried the offline Level 3 (L3) datasets available in Google Earth Engine (you can read more about the individual products here: https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p). For a given pollutant, for example NO2, we provide all data from the Sentinel 5P dataset for that pollutant. This includes the key measurements like NO2_column_number_density (a measure of NO2 concentration) as well as metadata like the satellite altitude. We recommend that you focus on the key measurements, either the column_number_density or the tropospheric_X_column_number_density (which measures density closer to Earth’s surface).
Unfortunately, this data is not 100% complete. Some locations have no sensor readings for a particular day, and so those rows have been excluded. There are also gaps in the input data, particularly the satellite data for CH4.

The Following data dictionary gives more details on this data set:

---

