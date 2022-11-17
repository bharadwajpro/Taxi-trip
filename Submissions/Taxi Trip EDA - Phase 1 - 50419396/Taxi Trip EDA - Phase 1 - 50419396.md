# Price analysis, Profitability and Affordability of Taxi trips in New York City

## Name: Venkata Krishna Bharadwaj Boinepally
## UB ID: 50419396
## UB Email: vboinepa@buffalo.edu
## Course: CSE 587: Data Intensive Computing
## Term: Fall 2022
## Project Name: [Phase 1] Price analysis, Profitability and Affordability of Taxi trips in New York City

## Motivation
- Modern taxi companies like Uber and Lyft have made the Taxi trip prices as a blackbox. No one other than the company know how the price is determined. It is important for passengers to know how much more the companies charge the passengers and make them aware of the factors that determine the price.
- Understanding the factors affecting the price brings more transparency and forces companies and consumers to make the best judgement as to what would be a fair price so that all the involved stackholders (Drivers, Riders and Companies) can sustain according to their best interest.
- Even though Uber provides some sort of per minute and per mile rates in their fare breakdown, the rates are variable according to time and location. Predictable pricing allows consumers to plan and adopt the taxi services more.

## Problem Statement
- The goal of this project is to do Exploratory Data Analysis (EDA) on the taxi trip dataset provided by the New York City's Taxi and Limousine Commision (TLC) and try to understand the patterns in each scenario and come to some well defined conclusions.
- These findings will help in coming up with the right Machine Learning model that tries to fit in the dataset.
- This ML Model can later be used to build a tool to show up the price estimate along with estimated share for each stackholder, given a pickup and dropoff location.

## Research Questions
1. Is the current form of pricing model adopted by Taxi companies benefit the consumers?
2. If a different model is to be adopted, what could be the possibilites which benefit all the stackholders like Drivers, Passengers and the Platform companies like Uber, Lyft?
3. Is there any way for the Taxi companies to remove the inefficiencies in the current business and pricing models and end up in profits? (For context: Uber and Lyft are still net loss making)

## Questions this project aims to answer
1. Understand the factors effecting price and how the price changes when we vary the factors
2. Analyse how expensive or inexpensive the taxi trips are, from the passenger point of view
3. Analyse how much profit/income the drivers get from each trip by deducting the expenses incurred
4. Analyse the profitability of the taxi companies like Uber, Lyft and why they are still in losses

## Why this is a significant problem?
- Uber and Lyft are dominating the taxi industry more than the traditional taxi companies.
    - These tech companies provide easy to use interface and on demand taxi service conveniently through app.
    - It's important to bring in some sort of transparency in the pricing models adopted by these tech companies.
    - This should make everyone aware of the factors determining the price and how it effects. And people can make the best judgement that suits them.
- This project findings will encourage further analysis in future and have the much needed conversations about pricing and business models take place.
- We can also gather insights about how the pricing algorithms changed over time by analysing historical trip data (not part of this project but possible future analysis)

## Data Sources
- The source of the data is from the New York City's Taxi & Limousine Commission (TLC) website - https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- TLC regularly updates their website with monthly datasets of all the taxi trips in NYC. Uber, Lyft, Yellow and Green taxis are part of this dataset.
- The website has historical trip data as well from 2013 to the current year 2022 separated on month to month basis.
- For the sake of simplicity and for starting the project, June 2022 data is being analysed for this project (in future phases, past data or other taxi companies trip data may be analysed).

## About this Notebook
- Throughout this notebook, you will find data cleaning and processing steps documented and explained properly.
- Similarly, you can find the corresponding EDA done and the results are documented near the cells.
- Conclusions and References are provided at the end.
- A report is provided along with this notebook which contains the information and results borrowed from this notebook in a brief format.

## Exploratory Data Analysis on Taxi Trip Data
(June 2022 Data, For Hire High Volume Vehicles only. Ex: Uber, Lyft)

Import numpy, pandas, matplotlib, seaborn


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
import seaborn as sns
```

Source of dataset [https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)


```python
# trip data
!wget https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-06.parquet
```

    --2022-10-13 12:02:25--  https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-06.parquet
    Resolving d37ci6vzurychx.cloudfront.net (d37ci6vzurychx.cloudfront.net)... 2600:9000:21dd:5600:b:20a5:b140:21, 2600:9000:21dd:c800:b:20a5:b140:21, 2600:9000:21dd:a600:b:20a5:b140:21, ...
    Connecting to d37ci6vzurychx.cloudfront.net (d37ci6vzurychx.cloudfront.net)|2600:9000:21dd:5600:b:20a5:b140:21|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 458193119 (437M) [application/x-www-form-urlencoded]
    Saving to: 'fhvhv_tripdata_2022-06.parquet.3'
    
    fhvhv_tripdata_2022 100%[===================>] 436.97M  11.4MB/s    in 54s     
    
    2022-10-13 12:03:20 (8.10 MB/s) - 'fhvhv_tripdata_2022-06.parquet.3' saved [458193119/458193119]
    



```python
# zone data
!wget https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv
```

    --2022-10-13 12:03:20--  https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv
    Resolving d37ci6vzurychx.cloudfront.net (d37ci6vzurychx.cloudfront.net)... 2600:9000:21dd:5600:b:20a5:b140:21, 2600:9000:21dd:c800:b:20a5:b140:21, 2600:9000:21dd:a600:b:20a5:b140:21, ...
    Connecting to d37ci6vzurychx.cloudfront.net (d37ci6vzurychx.cloudfront.net)|2600:9000:21dd:5600:b:20a5:b140:21|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 12322 (12K) [text/csv]
    Saving to: 'taxi+_zone_lookup.csv.3'
    
    taxi+_zone_lookup.c 100%[===================>]  12.03K  --.-KB/s    in 0s      
    
    2022-10-13 12:03:20 (60.9 MB/s) - 'taxi+_zone_lookup.csv.3' saved [12322/12322]
    


Do
1. pip install pyarrow
2. pip install fastparquet

For reading parquet files

Read trip data for June 2022


```python
df = pd.read_parquet('fhvhv_tripdata_2022-06.parquet')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hvfhs_license_num</th>
      <th>dispatching_base_num</th>
      <th>originating_base_num</th>
      <th>request_datetime</th>
      <th>on_scene_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_miles</th>
      <th>...</th>
      <th>sales_tax</th>
      <th>congestion_surcharge</th>
      <th>airport_fee</th>
      <th>tips</th>
      <th>driver_pay</th>
      <th>shared_request_flag</th>
      <th>shared_match_flag</th>
      <th>access_a_ride_flag</th>
      <th>wav_request_flag</th>
      <th>wav_match_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:20</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>234</td>
      <td>114</td>
      <td>1.500</td>
      <td>...</td>
      <td>0.68</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>9.36</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:40:36</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>161</td>
      <td>151</td>
      <td>4.180</td>
      <td>...</td>
      <td>1.81</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>4.82</td>
      <td>15.61</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:31:34</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>231</td>
      <td>87</td>
      <td>2.910</td>
      <td>...</td>
      <td>1.09</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.22</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:49:38</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>87</td>
      <td>225</td>
      <td>5.450</td>
      <td>...</td>
      <td>2.19</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>16.88</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HV0005</td>
      <td>B03406</td>
      <td>None</td>
      <td>2022-06-01 00:04:51</td>
      <td>NaT</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>137</td>
      <td>162</td>
      <td>1.069</td>
      <td>...</td>
      <td>0.73</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>5.47</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:23</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>74</td>
      <td>224</td>
      <td>6.070</td>
      <td>...</td>
      <td>1.56</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>2.24</td>
      <td>16.23</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:12</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>224</td>
      <td>13</td>
      <td>4.900</td>
      <td>...</td>
      <td>1.59</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>13.94</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:55:11</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>231</td>
      <td>231</td>
      <td>0.530</td>
      <td>...</td>
      <td>1.06</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>18.46</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>HV0003</td>
      <td>B03404</td>
      <td>B03404</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:04:58</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>234</td>
      <td>48</td>
      <td>2.850</td>
      <td>...</td>
      <td>2.79</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>27.27</td>
      <td>N</td>
      <td>N</td>
      <td></td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>HV0005</td>
      <td>B03406</td>
      <td>None</td>
      <td>2022-06-30 23:00:28</td>
      <td>NaT</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>244</td>
      <td>242</td>
      <td>6.207</td>
      <td>...</td>
      <td>1.80</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15.26</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>17780075 rows × 24 columns</p>
</div>



Read zone data


```python
zone_df = pd.read_csv('taxi+_zone_lookup.csv')
```


```python
zone_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Borough</th>
      <th>Zone</th>
      <th>service_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>EWR</td>
      <td>Newark Airport</td>
      <td>EWR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Queens</td>
      <td>Jamaica Bay</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bronx</td>
      <td>Allerton/Pelham Gardens</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Manhattan</td>
      <td>Alphabet City</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Staten Island</td>
      <td>Arden Heights</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>260</th>
      <td>261</td>
      <td>Manhattan</td>
      <td>World Trade Center</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>261</th>
      <td>262</td>
      <td>Manhattan</td>
      <td>Yorkville East</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>262</th>
      <td>263</td>
      <td>Manhattan</td>
      <td>Yorkville West</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>Unknown</td>
      <td>NV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>264</th>
      <td>265</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>265 rows × 4 columns</p>
</div>



Statistics for each column of trip data


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tolls</th>
      <th>bcf</th>
      <th>sales_tax</th>
      <th>congestion_surcharge</th>
      <th>airport_fee</th>
      <th>tips</th>
      <th>driver_pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
      <td>1.778008e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.390083e+02</td>
      <td>1.429608e+02</td>
      <td>5.169061e+00</td>
      <td>1.216133e+03</td>
      <td>2.507782e+01</td>
      <td>1.194475e+00</td>
      <td>7.946835e-01</td>
      <td>2.124014e+00</td>
      <td>1.160395e+00</td>
      <td>2.122203e-01</td>
      <td>1.164189e+00</td>
      <td>1.992252e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.492537e+01</td>
      <td>7.802175e+01</td>
      <td>6.058429e+00</td>
      <td>8.924502e+02</td>
      <td>2.111751e+01</td>
      <td>3.975068e+00</td>
      <td>7.077042e-01</td>
      <td>1.757838e+00</td>
      <td>1.366686e+00</td>
      <td>7.030242e-01</td>
      <td>3.247914e+00</td>
      <td>1.629283e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-1.055700e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-1.300000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.500000e+01</td>
      <td>7.600000e+01</td>
      <td>1.660000e+00</td>
      <td>6.180000e+02</td>
      <td>1.191000e+01</td>
      <td>0.000000e+00</td>
      <td>3.600000e-01</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>9.570000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000e+02</td>
      <td>1.420000e+02</td>
      <td>3.119000e+00</td>
      <td>9.840000e+02</td>
      <td>1.922000e+01</td>
      <td>0.000000e+00</td>
      <td>5.800000e-01</td>
      <td>1.640000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.536000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.110000e+02</td>
      <td>2.200000e+02</td>
      <td>6.420000e+00</td>
      <td>1.544000e+03</td>
      <td>3.084000e+01</td>
      <td>0.000000e+00</td>
      <td>9.700000e-01</td>
      <td>2.670000e+00</td>
      <td>2.750000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.477000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.650000e+02</td>
      <td>2.650000e+02</td>
      <td>6.259600e+02</td>
      <td>4.230900e+04</td>
      <td>2.409230e+03</td>
      <td>2.262000e+02</td>
      <td>7.291000e+01</td>
      <td>2.156800e+02</td>
      <td>1.100000e+01</td>
      <td>6.900000e+00</td>
      <td>1.500000e+02</td>
      <td>9.933100e+02</td>
    </tr>
  </tbody>
</table>
</div>



Column description for this data - [https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf)

### Data Cleaning/Processing - 1
- Remove unnecessary columns not relevant to our problem statement


```python
df.drop(columns=['dispatching_base_num',
                 'originating_base_num',
                 'on_scene_datetime',
                 'tolls',
                 'bcf',
                 'sales_tax',
                 'congestion_surcharge',
                 'airport_fee',
                 'shared_request_flag',
                 'shared_match_flag',
                 'access_a_ride_flag',
                 'wav_request_flag',
                 'wav_match_flag'],
        inplace=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hvfhs_license_num</th>
      <th>request_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tips</th>
      <th>driver_pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HV0003</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>234</td>
      <td>114</td>
      <td>1.500</td>
      <td>480</td>
      <td>7.68</td>
      <td>1.00</td>
      <td>9.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HV0003</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>161</td>
      <td>151</td>
      <td>4.180</td>
      <td>835</td>
      <td>20.40</td>
      <td>4.82</td>
      <td>15.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HV0003</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>231</td>
      <td>87</td>
      <td>2.910</td>
      <td>549</td>
      <td>12.29</td>
      <td>1.00</td>
      <td>8.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HV0003</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>87</td>
      <td>225</td>
      <td>5.450</td>
      <td>1197</td>
      <td>24.70</td>
      <td>0.00</td>
      <td>16.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HV0005</td>
      <td>2022-06-01 00:04:51</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>137</td>
      <td>162</td>
      <td>1.069</td>
      <td>234</td>
      <td>8.23</td>
      <td>0.00</td>
      <td>5.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>HV0003</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>74</td>
      <td>224</td>
      <td>6.070</td>
      <td>816</td>
      <td>17.60</td>
      <td>2.24</td>
      <td>16.23</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>HV0003</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>224</td>
      <td>13</td>
      <td>4.900</td>
      <td>710</td>
      <td>17.93</td>
      <td>0.00</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>HV0003</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>231</td>
      <td>231</td>
      <td>0.530</td>
      <td>595</td>
      <td>11.98</td>
      <td>3.00</td>
      <td>18.46</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>HV0003</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>234</td>
      <td>48</td>
      <td>2.850</td>
      <td>1184</td>
      <td>31.47</td>
      <td>1.00</td>
      <td>27.27</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>HV0005</td>
      <td>2022-06-30 23:00:28</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>244</td>
      <td>242</td>
      <td>6.207</td>
      <td>907</td>
      <td>20.28</td>
      <td>0.00</td>
      <td>15.26</td>
    </tr>
  </tbody>
</table>
<p>17780075 rows × 11 columns</p>
</div>



### Data Cleaning/Processing - 2
- Drop NA values


```python
df.dropna(inplace=True)
```

### Data Cleaning/Processing - 3
- Rename Columns and map the taxi company code to taxi name like Uber, Lyft

Actual codes
- HV0002: Juno
- HV0003: Uber
- HV0004: Via
- HV0005: Lyft


```python
df.rename(columns={
    'hvfhs_license_num': 'taxi_company',
    'PULocationID': 'PULocation',
    'DOLocationID': 'DOLocation'
}, inplace=True)
```


```python
def license_company_map(lic):
    if lic == 'HV0003':
        return 'Uber'
    elif lic == 'HV0005':
        return 'Lyft'
    return lic
```


```python
df['taxi_company'] = df['taxi_company'].map(license_company_map)
```

### Data Cleaning/Processing - 4
- Now there are some unknown locations in Pickup and Dropoff which we know by analysing Zone dataframe
- Remove the rows containing unknown locations in "df"


```python
zone_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Borough</th>
      <th>Zone</th>
      <th>service_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>EWR</td>
      <td>Newark Airport</td>
      <td>EWR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Queens</td>
      <td>Jamaica Bay</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bronx</td>
      <td>Allerton/Pelham Gardens</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Manhattan</td>
      <td>Alphabet City</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Staten Island</td>
      <td>Arden Heights</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>260</th>
      <td>261</td>
      <td>Manhattan</td>
      <td>World Trade Center</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>261</th>
      <td>262</td>
      <td>Manhattan</td>
      <td>Yorkville East</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>262</th>
      <td>263</td>
      <td>Manhattan</td>
      <td>Yorkville West</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>Unknown</td>
      <td>NV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>264</th>
      <td>265</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>265 rows × 4 columns</p>
</div>




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_company</th>
      <th>request_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocation</th>
      <th>DOLocation</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tips</th>
      <th>driver_pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uber</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>234</td>
      <td>114</td>
      <td>1.500</td>
      <td>480</td>
      <td>7.68</td>
      <td>1.00</td>
      <td>9.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Uber</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>161</td>
      <td>151</td>
      <td>4.180</td>
      <td>835</td>
      <td>20.40</td>
      <td>4.82</td>
      <td>15.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uber</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>231</td>
      <td>87</td>
      <td>2.910</td>
      <td>549</td>
      <td>12.29</td>
      <td>1.00</td>
      <td>8.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uber</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>87</td>
      <td>225</td>
      <td>5.450</td>
      <td>1197</td>
      <td>24.70</td>
      <td>0.00</td>
      <td>16.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lyft</td>
      <td>2022-06-01 00:04:51</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>137</td>
      <td>162</td>
      <td>1.069</td>
      <td>234</td>
      <td>8.23</td>
      <td>0.00</td>
      <td>5.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>Uber</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>74</td>
      <td>224</td>
      <td>6.070</td>
      <td>816</td>
      <td>17.60</td>
      <td>2.24</td>
      <td>16.23</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>Uber</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>224</td>
      <td>13</td>
      <td>4.900</td>
      <td>710</td>
      <td>17.93</td>
      <td>0.00</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>Uber</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>231</td>
      <td>231</td>
      <td>0.530</td>
      <td>595</td>
      <td>11.98</td>
      <td>3.00</td>
      <td>18.46</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>Uber</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>234</td>
      <td>48</td>
      <td>2.850</td>
      <td>1184</td>
      <td>31.47</td>
      <td>1.00</td>
      <td>27.27</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>Lyft</td>
      <td>2022-06-30 23:00:28</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>244</td>
      <td>242</td>
      <td>6.207</td>
      <td>907</td>
      <td>20.28</td>
      <td>0.00</td>
      <td>15.26</td>
    </tr>
  </tbody>
</table>
<p>17780075 rows × 11 columns</p>
</div>




```python
def unknown_locations(id):
    if id == 264 or id == 265:
        return True
    return False
```


```python
df[
    df.PULocation.map(unknown_locations)
].index
```




    Int64Index([   13254,    64649,    70100,    75130,    86298,    99255,
                  106944,   129350,   147399,   183654,
                ...
                17479230, 17500961, 17516820, 17545267, 17573284, 17618911,
                17700506, 17772055, 17773057, 17776276],
               dtype='int64', length=900)




```python
df.drop(df[
    df.PULocation.map(unknown_locations)
].index, inplace=True)
```


```python
df[
    df.DOLocation.map(unknown_locations)
].index
```




    Int64Index([      23,       26,       44,       68,       80,      138,
                     150,      168,      192,      240,
                ...
                17779915, 17779930, 17779977, 17779986, 17779990, 17779999,
                17780016, 17780024, 17780036, 17780049],
               dtype='int64', length=767811)




```python
df.drop(df[
    df.DOLocation.map(unknown_locations)
].index, inplace=True)
```

### Data Cleaning/Processing - 5
- There are some trips either having 0 miles as trip_miles or 0 seconds as time_time
- Remove those trips


```python
def zero_trip(trip_ms):
    if trip_ms == 0:
        return True
    return False
```


```python
df.drop(df[
    df.trip_miles.map(zero_trip)
].index, inplace=True)
```


```python
df.drop(df[
    df.trip_time.map(zero_trip)
].index, inplace=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_company</th>
      <th>request_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocation</th>
      <th>DOLocation</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tips</th>
      <th>driver_pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uber</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>234</td>
      <td>114</td>
      <td>1.500</td>
      <td>480</td>
      <td>7.68</td>
      <td>1.00</td>
      <td>9.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Uber</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>161</td>
      <td>151</td>
      <td>4.180</td>
      <td>835</td>
      <td>20.40</td>
      <td>4.82</td>
      <td>15.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uber</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>231</td>
      <td>87</td>
      <td>2.910</td>
      <td>549</td>
      <td>12.29</td>
      <td>1.00</td>
      <td>8.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uber</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>87</td>
      <td>225</td>
      <td>5.450</td>
      <td>1197</td>
      <td>24.70</td>
      <td>0.00</td>
      <td>16.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lyft</td>
      <td>2022-06-01 00:04:51</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>137</td>
      <td>162</td>
      <td>1.069</td>
      <td>234</td>
      <td>8.23</td>
      <td>0.00</td>
      <td>5.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>Uber</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>74</td>
      <td>224</td>
      <td>6.070</td>
      <td>816</td>
      <td>17.60</td>
      <td>2.24</td>
      <td>16.23</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>Uber</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>224</td>
      <td>13</td>
      <td>4.900</td>
      <td>710</td>
      <td>17.93</td>
      <td>0.00</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>Uber</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>231</td>
      <td>231</td>
      <td>0.530</td>
      <td>595</td>
      <td>11.98</td>
      <td>3.00</td>
      <td>18.46</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>Uber</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>234</td>
      <td>48</td>
      <td>2.850</td>
      <td>1184</td>
      <td>31.47</td>
      <td>1.00</td>
      <td>27.27</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>Lyft</td>
      <td>2022-06-30 23:00:28</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>244</td>
      <td>242</td>
      <td>6.207</td>
      <td>907</td>
      <td>20.28</td>
      <td>0.00</td>
      <td>15.26</td>
    </tr>
  </tbody>
</table>
<p>17007449 rows × 11 columns</p>
</div>




```python
zone_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Borough</th>
      <th>Zone</th>
      <th>service_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>EWR</td>
      <td>Newark Airport</td>
      <td>EWR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Queens</td>
      <td>Jamaica Bay</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bronx</td>
      <td>Allerton/Pelham Gardens</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Manhattan</td>
      <td>Alphabet City</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Staten Island</td>
      <td>Arden Heights</td>
      <td>Boro Zone</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>260</th>
      <td>261</td>
      <td>Manhattan</td>
      <td>World Trade Center</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>261</th>
      <td>262</td>
      <td>Manhattan</td>
      <td>Yorkville East</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>262</th>
      <td>263</td>
      <td>Manhattan</td>
      <td>Yorkville West</td>
      <td>Yellow Zone</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>Unknown</td>
      <td>NV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>264</th>
      <td>265</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>265 rows × 4 columns</p>
</div>



### Data Cleaning/Processing - 6
- Expand the pickup and dropoff column to contain location names instead of location code
- This is used in later phases to lookup addresses and estimate distance


```python
zone_df['LocationName'] = zone_df['Zone'] + ", " + zone_df['Borough']
```


```python
zone_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LocationID</th>
      <th>Borough</th>
      <th>Zone</th>
      <th>service_zone</th>
      <th>LocationName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>EWR</td>
      <td>Newark Airport</td>
      <td>EWR</td>
      <td>Newark Airport, EWR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Queens</td>
      <td>Jamaica Bay</td>
      <td>Boro Zone</td>
      <td>Jamaica Bay, Queens</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bronx</td>
      <td>Allerton/Pelham Gardens</td>
      <td>Boro Zone</td>
      <td>Allerton/Pelham Gardens, Bronx</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Manhattan</td>
      <td>Alphabet City</td>
      <td>Yellow Zone</td>
      <td>Alphabet City, Manhattan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Staten Island</td>
      <td>Arden Heights</td>
      <td>Boro Zone</td>
      <td>Arden Heights, Staten Island</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>260</th>
      <td>261</td>
      <td>Manhattan</td>
      <td>World Trade Center</td>
      <td>Yellow Zone</td>
      <td>World Trade Center, Manhattan</td>
    </tr>
    <tr>
      <th>261</th>
      <td>262</td>
      <td>Manhattan</td>
      <td>Yorkville East</td>
      <td>Yellow Zone</td>
      <td>Yorkville East, Manhattan</td>
    </tr>
    <tr>
      <th>262</th>
      <td>263</td>
      <td>Manhattan</td>
      <td>Yorkville West</td>
      <td>Yellow Zone</td>
      <td>Yorkville West, Manhattan</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>Unknown</td>
      <td>NV</td>
      <td>NaN</td>
      <td>NV, Unknown</td>
    </tr>
    <tr>
      <th>264</th>
      <td>265</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>265 rows × 5 columns</p>
</div>




```python
zone_df.set_index('LocationID', inplace=True)
```


```python
id_location_map = zone_df['LocationName'].T.to_dict()
```


```python
df['PULocation'] = df['PULocation'].map(id_location_map)
```


```python
df['DOLocation'] = df['DOLocation'].map(id_location_map)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_company</th>
      <th>request_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocation</th>
      <th>DOLocation</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tips</th>
      <th>driver_pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uber</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>Union Sq, Manhattan</td>
      <td>Greenwich Village South, Manhattan</td>
      <td>1.500</td>
      <td>480</td>
      <td>7.68</td>
      <td>1.00</td>
      <td>9.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Uber</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>Midtown Center, Manhattan</td>
      <td>Manhattan Valley, Manhattan</td>
      <td>4.180</td>
      <td>835</td>
      <td>20.40</td>
      <td>4.82</td>
      <td>15.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uber</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>Financial District North, Manhattan</td>
      <td>2.910</td>
      <td>549</td>
      <td>12.29</td>
      <td>1.00</td>
      <td>8.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uber</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>Financial District North, Manhattan</td>
      <td>Stuyvesant Heights, Brooklyn</td>
      <td>5.450</td>
      <td>1197</td>
      <td>24.70</td>
      <td>0.00</td>
      <td>16.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lyft</td>
      <td>2022-06-01 00:04:51</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>Kips Bay, Manhattan</td>
      <td>Midtown East, Manhattan</td>
      <td>1.069</td>
      <td>234</td>
      <td>8.23</td>
      <td>0.00</td>
      <td>5.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>Uber</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>East Harlem North, Manhattan</td>
      <td>Stuy Town/Peter Cooper Village, Manhattan</td>
      <td>6.070</td>
      <td>816</td>
      <td>17.60</td>
      <td>2.24</td>
      <td>16.23</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>Uber</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>Stuy Town/Peter Cooper Village, Manhattan</td>
      <td>Battery Park City, Manhattan</td>
      <td>4.900</td>
      <td>710</td>
      <td>17.93</td>
      <td>0.00</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>Uber</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>0.530</td>
      <td>595</td>
      <td>11.98</td>
      <td>3.00</td>
      <td>18.46</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>Uber</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>Union Sq, Manhattan</td>
      <td>Clinton East, Manhattan</td>
      <td>2.850</td>
      <td>1184</td>
      <td>31.47</td>
      <td>1.00</td>
      <td>27.27</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>Lyft</td>
      <td>2022-06-30 23:00:28</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>Washington Heights South, Manhattan</td>
      <td>Van Nest/Morris Park, Bronx</td>
      <td>6.207</td>
      <td>907</td>
      <td>20.28</td>
      <td>0.00</td>
      <td>15.26</td>
    </tr>
  </tbody>
</table>
<p>17007449 rows × 11 columns</p>
</div>



### Data Cleaning/Processing - 7
- Each trip has different number of miles, time taken and the final price
- To compare the prices among different trips, a new column called "fare_per_mile_per_second" is created which is obtained by dividing the "base_passenger_fare" by the "trip_miles" and "trip_time"


```python
df['fare_per_mile_per_second'] = df['base_passenger_fare']/(df['trip_miles']*df['trip_time'])
```

### Data Cleaning/Processing - 8
- Create a new dataframe grouped by "taxi_company".
- This will be used to perform various analysis and compare results between different taxi companies. (Uber and Lyft)


```python
taxi_grp_df = df.groupby(['taxi_company'])
```

"plot_bar" function to plot bar chart


```python
def plot_bar(obj):
    obj.plot(kind='bar', title=obj.name, legend=True)
```

### Exploratory Data Analysis (EDA) - 1

Number of trips by each taxi company
- Uber has the largest amount of trips close to 12.4M, while Lyft has 4.5M trips in June 2022
- Uber accounted for 73% of the total trips and Lyft has remaining 27% of the trips
- Each taxi company have different models to determine the fare. So it's important to separate them and analyse.
- We can also find the average price by each taxi company.
    - Uber seems cheaper than Lyft by about $0.04 per mile per second.


```python
df['taxi_company'].value_counts()
```




    Uber    12473021
    Lyft     4534428
    Name: taxi_company, dtype: int64




```python
df['taxi_company'].value_counts(normalize=True)
```




    Uber    0.733386
    Lyft    0.266614
    Name: taxi_company, dtype: float64




```python
plot_bar(df['taxi_company'].value_counts())
```


    
![png](output_67_0.png)
    



```python
plot_bar(taxi_grp_df['fare_per_mile_per_second'].mean())
```


    
![png](output_68_0.png)
    


### Data Cleaning/Processing - 9
- Extract the hour from the given date time column for Request, Pickup and Dropoff
- This is used to analyse the trips during different times of day


```python
df['request_hour'] = df['request_datetime'].dt.hour
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
```

### Exploratory Data Analysis (EDA) - 2
- Plotting the number of trips that happened during each hour of the day.
- Time between 16:00 to 23:59 is the busiest of the day with each hour accounting for greater than 5% of the trips.
- Time between 2:00 to 5:59 is the least busy hours with less than 1% of the trips.
- This trend is observed in all the "request", "pickup" and "dropoff" hours
- Taxi companies tend to charge "Surcharge" during busy times or when the supply of drivers is less. So it's important to identify the hour of the day.


```python
df['request_hour'].value_counts(normalize=True)
```




    18    0.058690
    17    0.056377
    22    0.055772
    19    0.055551
    21    0.054995
    20    0.052570
    16    0.051573
    23    0.050122
    15    0.049752
    14    0.048433
    8     0.047282
    9     0.045285
    13    0.044727
    12    0.043822
    11    0.043298
    10    0.042687
    7     0.038602
    0     0.037971
    1     0.027288
    6     0.026877
    2     0.019852
    5     0.017952
    3     0.015586
    4     0.014937
    Name: request_hour, dtype: float64




```python
plot_bar(df['request_hour'].value_counts())
```


    
![png](output_73_0.png)
    



```python
plot_bar(df['pickup_hour'].value_counts())
```


    
![png](output_74_0.png)
    



```python
plot_bar(df['dropoff_hour'].value_counts())
```


    
![png](output_75_0.png)
    


### Exploratory Data Analysis (EDA) - 3
- Top 20 Popular Pickup locations/zones. These account for more than 1% of all trips and together more than 20%.
- Similarly, Top 20 Popular Dropoff locations.
- Most of these locations are Airports, Tourist destinations.
- Popular locations tend to have more users requesting for taxi. This can increase the price.


```python
df['PULocation'].value_counts(normalize=True)[:20]
```




    LaGuardia Airport, Queens               0.018295
    JFK Airport, Queens                     0.015080
    East Village, Manhattan                 0.014320
    Times Sq/Theatre District, Manhattan    0.013628
    Crown Heights North, Brooklyn           0.013489
    East Chelsea, Manhattan                 0.012561
    Midtown Center, Manhattan               0.012506
    TriBeCa/Civic Center, Manhattan         0.012282
    Bushwick South, Brooklyn                0.011677
    West Chelsea/Hudson Yards, Manhattan    0.011441
    Lower East Side, Manhattan              0.011429
    Union Sq, Manhattan                     0.011098
    Clinton East, Manhattan                 0.010927
    Midtown South, Manhattan                0.010765
    East New York, Brooklyn                 0.010574
    Williamsburg (North Side), Brooklyn     0.010475
    West Village, Manhattan                 0.010193
    Astoria, Queens                         0.010111
    Murray Hill, Manhattan                  0.010027
    Park Slope, Brooklyn                    0.009892
    Name: PULocation, dtype: float64




```python
sum(df['PULocation'].value_counts(normalize=True)[:20])
```




    0.24076985325665243




```python
plot_bar(df['PULocation'].value_counts()[:20])
```


    
![png](output_79_0.png)
    



```python
plot_bar(df['DOLocation'].value_counts()[:20])
```


    
![png](output_80_0.png)
    


Graph plotting helper function to display histogram


```python
def plot_hist(df, series, remove_extremes=True):
    max_value = df[series].quantile(0.99)
    if remove_extremes:
        df = df[df[series] < max_value]
        df = df[df[series] > 0]
    df[series].hist(bins=range(int(max_value)), legend=True)
```

### Exploratory Data Analysis (EDA) - 4
- Plotting the time of each trip as a histogram.
- From the plot, we can see that most of the trips' duration tend to be less than 1000 seconds (15 minutes approximately)
- So the taxi drivers and passengers prefer to have short trips throughout the day.
- This can effect the final fare when a passenger requests for a longer trip (let's say > 30 minutes)
    - Conclusion from the analysis below is, shorter trips tend to be expensive than longer trips
    - Shorter trips are 20 times more expensive than longer trips on average.


```python
plot_hist(df, 'trip_time')
```


    
![png](output_84_0.png)
    



```python
longer_trip_df = df.copy()[df['trip_time'] > 30*60]
shorter_trip_df = df.copy()[df['trip_time'] < 15*60]
```


```python
longer_trip_df['fare_per_mile_per_second'].mean()
```




    0.00204712666870961




```python
shorter_trip_df['fare_per_mile_per_second'].mean()
```




    0.041748159149506046



### Exploratory Data Analysis (EDA) - 5
- Plotting the absolute value of "base_passenger_fare" against the number of trips as a histogram
- This tells us that most of the trips have "Base Passenger Fare" as less than \\$10 with peak at \\$5
- Similarly plotting the Tips histogram and Driver Pay histogram reveals these results
    - Most of the tips are in the range of \\$2-\\$5
    - Driver pay for most trips is in the range of \\$8-\\$10


```python
plot_hist(df, 'base_passenger_fare')
```


    
![png](output_89_0.png)
    



```python
plot_hist(df, 'tips')
```


    
![png](output_90_0.png)
    



```python
plot_hist(df, 'driver_pay')
```


    
![png](output_91_0.png)
    


### Data Cleaning/Processing - 10
- Create a new column "driver_percent_in_fare" to analyse the driver's pay as a percentage of the "base_passenger_fare"
- This is used later to understand how much of a profit margin Taxi companies are targetting to achieve. This profit margin directly effects the price passengers pay.

### Exploratory Data Analysis (EDA) - 6
- A portion of the base passenger fare goes to the driver and the remaining goes to the taxi company
    - Note that other taxes and fees such as toll, airport fee and taxes are charged to the passenger in addition to the base fare
- Here, we can see how much portion of the fare is paid to the driver
- Analysis show that approximately 80% of the fare is paid to the driver with most trips having 60% to 100% paid to driver


```python
df['driver_percent_in_fare'] = df['driver_pay']/df['base_passenger_fare']*100
plot_hist(df, 'driver_percent_in_fare')
```


    
![png](output_94_0.png)
    


Seaborn plot utility functions for histogram and bar charts (Used for plotting grouped by data)


```python
def plot_sns_grp_hist(grp_df, x, hue):
    ax = sns.histplot(grp_df, x=x, hue=hue, multiple='stack')
    ax.set(xlim=(0, grp_df[x].quantile(0.99)))
```


```python
def plot_sns_grp_bar(grp_df, x, hue):
    ax = sns.barplot(grp_df, x=x, hue=hue, multiple='stack')
    ax.set(xlim=(0, grp_df[x].quantile(0.99)))
```

### Exploratory Data Analysis (EDA) - 7
- General analysis on grouped data on which taxi company users tip more, driver pay, passenger fare and user preference according to the company

#### Which users tip more? (Uber or Lyft)
Lyft


```python
taxi_grp_df = df.groupby(['taxi_company'])
```


```python
taxi_grp_df['tips'].mean()
```




    taxi_company
    Lyft    1.142484
    Uber    1.032130
    Name: tips, dtype: float64




```python
plot_sns_grp_hist(df, 'tips', 'taxi_company')
```


    
![png](output_102_0.png)
    


#### Which drivers earn more? (Uber or Lyft)
Uber


```python
taxi_grp_df['driver_pay'].mean()
```




    taxi_company
    Lyft    17.249506
    Uber    19.016435
    Name: driver_pay, dtype: float64




```python
plot_sns_grp_hist(df, 'driver_pay', 'taxi_company')
```


    
![png](output_105_0.png)
    


#### Which is cheaper for passenger?
Lyft


```python
taxi_grp_df['base_passenger_fare'].mean()
```




    taxi_company
    Lyft    22.561838
    Uber    23.572609
    Name: base_passenger_fare, dtype: float64



#### Preference for longer trips
Lyft


```python
taxi_grp_df['trip_miles'].mean()
```




    taxi_company
    Lyft    4.677353
    Uber    4.579548
    Name: trip_miles, dtype: float64




```python
plot_sns_grp_hist(df, 'trip_miles', 'taxi_company')
```


    
![png](output_110_0.png)
    


#### Passenger preference for "Uber vs Lyft" according to different times of day


```python
plot_sns_grp_hist(df, 'pickup_hour', 'taxi_company')
```


    
![png](output_112_0.png)
    



```python
plot_sns_grp_hist(df, 'dropoff_hour', 'taxi_company')
```


    
![png](output_113_0.png)
    


Closely correlated and no specific preference

Utility functions to remove outliers, normalize column data using mean-std.dev or min-max procedures


```python
def remove_outliers(df, column):
    min_value = df[column].quantile(0.01)
    max_value = df[column].quantile(0.99)
    df = df[df[column] < max_value]
    df = df[df[column] > min_value]
```


```python
def mean_std_norm(df, column):
    meanv = df[column].mean()
    stdv = df[column].std()
    df[column] = (df[column] - meanv)/stdv
    df[column] += 1
    df[column] /= 2
```


```python
def min_max_norm(df, column):
    minv = df[column].min()
    maxv = df[column].max()
    df[column] = 100*(df[column] - minv)/(maxv - minv)
```


```python
# Used later
# remove_outliers(df, 'fare_per_mile_per_second')
# min_max_norm(df, 'fare_per_mile_per_second')
```

### Exploratory Data Analysis (EDA) - 8
- Comparing the fare prices during different times of day
- At 05:00, the fare is highest, probably because of very less drivers during that time.
    - This is confirmed when we plot the number of trips during the 02:00 to 05:00 hours.
- High prices are observed during 02:00 to 05:00
- Lowest fares during 09:00 to 11:00 and 19:00 to 20:00


```python
grp_by_pickup_hr = df.groupby(['pickup_hour'])
fms_by_hr = grp_by_pickup_hr['fare_per_mile_per_second'].mean()
plot_bar(fms_by_hr)
```


    
![png](output_121_0.png)
    



```python
plot_bar(grp_by_pickup_hr.size())
```


    
![png](output_122_0.png)
    



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_company</th>
      <th>request_datetime</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>PULocation</th>
      <th>DOLocation</th>
      <th>trip_miles</th>
      <th>trip_time</th>
      <th>base_passenger_fare</th>
      <th>tips</th>
      <th>driver_pay</th>
      <th>fare_per_mile_per_second</th>
      <th>request_hour</th>
      <th>pickup_hour</th>
      <th>dropoff_hour</th>
      <th>driver_percent_in_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uber</td>
      <td>2022-06-01 00:15:35</td>
      <td>2022-06-01 00:17:41</td>
      <td>2022-06-01 00:25:41</td>
      <td>Union Sq, Manhattan</td>
      <td>Greenwich Village South, Manhattan</td>
      <td>1.500</td>
      <td>480</td>
      <td>7.68</td>
      <td>1.00</td>
      <td>9.36</td>
      <td>0.010667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>121.875000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Uber</td>
      <td>2022-06-01 00:39:04</td>
      <td>2022-06-01 00:42:37</td>
      <td>2022-06-01 00:56:32</td>
      <td>Midtown Center, Manhattan</td>
      <td>Manhattan Valley, Manhattan</td>
      <td>4.180</td>
      <td>835</td>
      <td>20.40</td>
      <td>4.82</td>
      <td>15.61</td>
      <td>0.005845</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>76.519608</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uber</td>
      <td>2022-06-01 00:27:53</td>
      <td>2022-06-01 00:36:22</td>
      <td>2022-06-01 00:45:31</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>Financial District North, Manhattan</td>
      <td>2.910</td>
      <td>549</td>
      <td>12.29</td>
      <td>1.00</td>
      <td>8.22</td>
      <td>0.007693</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>66.883645</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uber</td>
      <td>2022-06-01 00:48:15</td>
      <td>2022-06-01 00:51:18</td>
      <td>2022-06-01 01:11:15</td>
      <td>Financial District North, Manhattan</td>
      <td>Stuyvesant Heights, Brooklyn</td>
      <td>5.450</td>
      <td>1197</td>
      <td>24.70</td>
      <td>0.00</td>
      <td>16.88</td>
      <td>0.003786</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>68.340081</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lyft</td>
      <td>2022-06-01 00:04:51</td>
      <td>2022-06-01 00:13:33</td>
      <td>2022-06-01 00:17:27</td>
      <td>Kips Bay, Manhattan</td>
      <td>Midtown East, Manhattan</td>
      <td>1.069</td>
      <td>234</td>
      <td>8.23</td>
      <td>0.00</td>
      <td>5.47</td>
      <td>0.032901</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>66.464156</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17780070</th>
      <td>Uber</td>
      <td>2022-06-30 23:20:49</td>
      <td>2022-06-30 23:24:43</td>
      <td>2022-06-30 23:38:19</td>
      <td>East Harlem North, Manhattan</td>
      <td>Stuy Town/Peter Cooper Village, Manhattan</td>
      <td>6.070</td>
      <td>816</td>
      <td>17.60</td>
      <td>2.24</td>
      <td>16.23</td>
      <td>0.003553</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>92.215909</td>
    </tr>
    <tr>
      <th>17780071</th>
      <td>Uber</td>
      <td>2022-06-30 23:36:13</td>
      <td>2022-06-30 23:39:20</td>
      <td>2022-06-30 23:51:10</td>
      <td>Stuy Town/Peter Cooper Village, Manhattan</td>
      <td>Battery Park City, Manhattan</td>
      <td>4.900</td>
      <td>710</td>
      <td>17.93</td>
      <td>0.00</td>
      <td>13.94</td>
      <td>0.005154</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>77.746793</td>
    </tr>
    <tr>
      <th>17780072</th>
      <td>Uber</td>
      <td>2022-06-30 23:50:50</td>
      <td>2022-06-30 23:57:12</td>
      <td>2022-07-01 00:07:07</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>TriBeCa/Civic Center, Manhattan</td>
      <td>0.530</td>
      <td>595</td>
      <td>11.98</td>
      <td>3.00</td>
      <td>18.46</td>
      <td>0.037990</td>
      <td>23</td>
      <td>23</td>
      <td>0</td>
      <td>154.090150</td>
    </tr>
    <tr>
      <th>17780073</th>
      <td>Uber</td>
      <td>2022-06-30 23:02:40</td>
      <td>2022-06-30 23:06:44</td>
      <td>2022-06-30 23:26:28</td>
      <td>Union Sq, Manhattan</td>
      <td>Clinton East, Manhattan</td>
      <td>2.850</td>
      <td>1184</td>
      <td>31.47</td>
      <td>1.00</td>
      <td>27.27</td>
      <td>0.009326</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>86.653956</td>
    </tr>
    <tr>
      <th>17780074</th>
      <td>Lyft</td>
      <td>2022-06-30 23:00:28</td>
      <td>2022-06-30 23:03:06</td>
      <td>2022-06-30 23:18:13</td>
      <td>Washington Heights South, Manhattan</td>
      <td>Van Nest/Morris Park, Bronx</td>
      <td>6.207</td>
      <td>907</td>
      <td>20.28</td>
      <td>0.00</td>
      <td>15.26</td>
      <td>0.003602</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>75.246548</td>
    </tr>
  </tbody>
</table>
<p>17007449 rows × 16 columns</p>
</div>



### Exploratory Data Analysis (EDA) - 9
- Plot the most and least expensive pickup and dropoff locations
- The ones with higher pricing tend to be of places where the number of trips are very high (10M vs 175K). The demand for taxi is more, hence increased fare. Vice versa for least expensive locations.
- Similar results are observed when analysing dropoff locations


```python
grp_by_pickup_location = df.groupby(['PULocation'])
sorted_pickup_locations = grp_by_pickup_location['fare_per_mile_per_second'].mean().sort_values(ascending=False)
```


```python
expensive_pickup_locations = sorted_pickup_locations[:20]
plot_bar(expensive_pickup_locations)
```


    
![png](output_126_0.png)
    



```python
cheap_pickup_locations = sorted_pickup_locations[-20:]
plot_bar(cheap_pickup_locations)
```


    
![png](output_127_0.png)
    



```python
# Expensive pickup locations' mean fare
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[:20])]['fare_per_mile_per_second'].mean()
```




    0.07918098264343432




```python
# Inexpensive pickup locations' mean fare
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[-20:])]['fare_per_mile_per_second'].mean()
```




    0.00839567837002391



Almost 10x the fare of least expensive locations


```python
# Number of trips in Expensive pickup locations
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[:20])].shape[0]
```




    535649




```python
# Number of trips in Inexpensive pickup locations
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[-20:])].shape[0]
```




    175107



Close to 5x the number of trips in inexpensive areas


```python
grp_by_dropoff_location = df.groupby(['DOLocation'])
sorted_dropoff_locations = grp_by_dropoff_location['fare_per_mile_per_second'].mean().sort_values(ascending=False)
```


```python
expensive_dropoff_locations = sorted_dropoff_locations[:20]
plot_bar(expensive_dropoff_locations)
```


    
![png](output_135_0.png)
    



```python
cheap_dropoff_locations = sorted_dropoff_locations[-20:]
plot_bar(cheap_dropoff_locations)
```


    
![png](output_136_0.png)
    



```python
# Expensive dropoff locations' mean fare
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[:20])]['fare_per_mile_per_second'].mean()
```




    0.0723166739043836




```python
# Inexpensive dropoff locations' mean fare
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[-20:])]['fare_per_mile_per_second'].mean()
```




    0.006250708746235113



Almost 10x the fare of least expensive locations


```python
# Number of trips in Expensive dropoff locations
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[:20])].shape[0]
```




    789883




```python
# Number of trips in Inexpensive dropoff locations
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[-20:])].shape[0]
```




    417824



More than 2x the number of trips in inexpensive areas

### Exploratory Data Analysis (EDA) - 10
- Plot the average fare for popular pickup and dropoff locations
- Most of these locations are Airports, Tourist destinations
- Among these locations, some locations have average fare of \\$0.01 per mile per second whereas some have \\$0.04 per mile per second (almost 4x times the least one)
- Popular dropoff locations also follow similar pattern with fare ranging from 1x-6x


```python
popular_pickup_locations = grp_by_pickup_location.size().sort_values(ascending=False).reset_index()
```


```python
plot_bar(grp_by_pickup_location['fare_per_mile_per_second'].mean()[popular_pickup_locations['PULocation']][:20])
```


    
![png](output_145_0.png)
    



```python
popular_dropoff_locations = grp_by_dropoff_location.size().sort_values(ascending=False).reset_index()
```


```python
plot_bar(grp_by_dropoff_location['fare_per_mile_per_second'].mean()[popular_dropoff_locations['DOLocation']][:20])
```


    
![png](output_147_0.png)
    


### Data Cleaning/Processing - 11
- In order to understand the true cost of each trip and the driver profits after deducting their gas cost, new columns like "driver_expenditure", "driver_profit" and "taxi_company_profit" are created.
- This is used in later phases to predict the base fare and how little or more the taxi company can charge the passenger and still end up in profit.
- Average mileage and gas price for the month are hyperparameters. We can choose the value according to the car type and gas prices


```python
mpg = 20 # average miles per gallon
pg = 5.56 # NYC gas price in June 2022
df['driver_expenditure'] = (df['trip_miles']/mpg)*pg
df['driver_profit'] = df['driver_pay'] - df['driver_expenditure']
df['taxi_company_profit'] = df['base_passenger_fare'] - df['driver_pay']
```

### Exploratory Data Analysis (EDA) - 11
- Plotting the "Driver's Profit" after deducting the expenses such as gas
- Plotting "Taxi Companies' Profit" after removing Driver's share from the Base passenger fare
- Driver's mostly get paid \\$5-\\$8 per trip with more pay for longer trips
- Taxi Companies profit is within a range for most trips with the range being \\$1-\\$5 (more pay for longer trips)


```python
plot_hist(df, 'driver_profit')
```


    
![png](output_151_0.png)
    



```python
plot_hist(df, 'taxi_company_profit')
```


    
![png](output_152_0.png)
    


## Conclusion
- All the findings for each EDA are listed above, near each cell
- Trip time, trip miles, pickup location, dropoff location, driver's profit, taxi company profit margin effect the base passenger fare
- Other factors like tips also effect final price
- All of these factors will be taken into consideration when developing the Machine Learning models in Phase 2

## References
- https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- https://registry.opendata.aws/nyc-tlc-trip-records-pds/
- Seaborn, Pandas, Numpy documentation
