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

Do
1. pip install pyarrow
2. pip install fastparquet

For reading parquet files

### Data Cleaning/Processing - 1
- Remove unnecessary columns not relevant to our problem statement

### Data Cleaning/Processing - 2
- Drop NA values

### Data Cleaning/Processing - 3
- Rename Columns and map the taxi company code to taxi name like Uber, Lyft

Actual codes
- HV0002: Juno
- HV0003: Uber
- HV0004: Via
- HV0005: Lyft

### Data Cleaning/Processing - 4
- Now there are some unknown locations in Pickup and Dropoff which we know by analysing Zone dataframe
- Remove the rows containing unknown locations in "df"

### Data Cleaning/Processing - 5
- There are some trips either having 0 miles as trip_miles or 0 seconds as time_time
- Remove those trips

### Data Cleaning/Processing - 6
- Expand the pickup and dropoff column to contain location names instead of location code
- This is used in later phases to lookup addresses and estimate distance

### Data Cleaning/Processing - 7
- Each trip has different number of miles, time taken and the final price
- To compare the prices among different trips, a new column called "fare_per_mile_per_second" is created which is obtained by dividing the "base_passenger_fare" by the "trip_miles" and "trip_time"

### Data Cleaning/Processing - 8
- Create a new dataframe grouped by "taxi_company".
- This will be used to perform various analysis and compare results between different taxi companies. (Uber and Lyft)

### Exploratory Data Analysis (EDA) - 1

Number of trips by each taxi company
- Uber has the largest amount of trips close to 12.4M, while Lyft has 4.5M trips in June 2022
- Uber accounted for 73% of the total trips and Lyft has remaining 27% of the trips
- Each taxi company have different models to determine the fare. So it's important to separate them and analyse.
- We can also find the average price by each taxi company.
    - Uber seems cheaper than Lyft by about $0.04 per mile per second.

### Data Cleaning/Processing - 9
- Extract the hour from the given date time column for Request, Pickup and Dropoff
- This is used to analyse the trips during different times of day

### Exploratory Data Analysis (EDA) - 2
- Plotting the number of trips that happened during each hour of the day.
- Time between 16:00 to 23:59 is the busiest of the day with each hour accounting for greater than 5% of the trips.
- Time between 2:00 to 5:59 is the least busy hours with less than 1% of the trips.
- This trend is observed in all the "request", "pickup" and "dropoff" hours
- Taxi companies tend to charge "Surcharge" during busy times or when the supply of drivers is less. So it's important to identify the hour of the day.

### Exploratory Data Analysis (EDA) - 3
- Top 20 Popular Pickup locations/zones. These account for more than 1% of all trips and together more than 20%.
- Similarly, Top 20 Popular Dropoff locations.
- Most of these locations are Airports, Tourist destinations.
- Popular locations tend to have more users requesting for taxi. This can increase the price.

### Exploratory Data Analysis (EDA) - 4
- Plotting the time of each trip as a histogram.
- From the plot, we can see that most of the trips' duration tend to be less than 1000 seconds (15 minutes approximately)
- So the taxi drivers and passengers prefer to have short trips throughout the day.
- This can effect the final fare when a passenger requests for a longer trip (let's say > 30 minutes)
    - Conclusion from the analysis below is, shorter trips tend to be expensive than longer trips
    - Shorter trips are 20 times more expensive than longer trips on average.

### Exploratory Data Analysis (EDA) - 5
- Plotting the absolute value of "base_passenger_fare" against the number of trips as a histogram
- This tells us that most of the trips have "Base Passenger Fare" as less than \\$10 with peak at \\$5
- Similarly plotting the Tips histogram and Driver Pay histogram reveals these results
    - Most of the tips are in the range of \\$2-\\$5
    - Driver pay for most trips is in the range of \\$8-\\$10

### Data Cleaning/Processing - 10
- Create a new column "driver_percent_in_fare" to analyse the driver's pay as a percentage of the "base_passenger_fare"
- This is used later to understand how much of a profit margin Taxi companies are targetting to achieve. This profit margin directly effects the price passengers pay.

### Exploratory Data Analysis (EDA) - 6
- A portion of the base passenger fare goes to the driver and the remaining goes to the taxi company
    - Note that other taxes and fees such as toll, airport fee and taxes are charged to the passenger in addition to the base fare
- Here, we can see how much portion of the fare is paid to the driver
- Analysis show that approximately 80% of the fare is paid to the driver with most trips having 60% to 100% paid to driver

### Exploratory Data Analysis (EDA) - 7
- General analysis on grouped data on which taxi company users tip more, driver pay, passenger fare and user preference according to the company

#### Which users tip more? (Uber or Lyft)
Lyft

#### Which drivers earn more? (Uber or Lyft)
Uber

#### Which is cheaper for passenger?
Lyft

#### Preference for longer trips
Lyft

#### Passenger preference for "Uber vs Lyft" according to different times of day

Closely correlated and no specific preference

### Exploratory Data Analysis (EDA) - 8
- Comparing the fare prices during different times of day
- At 05:00, the fare is highest, probably because of very less drivers during that time.
    - This is confirmed when we plot the number of trips during the 02:00 to 05:00 hours.
- High prices are observed during 02:00 to 05:00
- Lowest fares during 09:00 to 11:00 and 19:00 to 20:00

### Exploratory Data Analysis (EDA) - 9
- Plot the most and least expensive pickup and dropoff locations
- The ones with higher pricing tend to be of places where the number of trips are very high (10M vs 175K). The demand for taxi is more, hence increased fare. Vice versa for least expensive locations.
- Similar results are observed when analysing dropoff locations

#### Expensive pickup locations' mean fare
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[:20])]['fare_per_mile_per_second'].mean()

#### Inexpensive pickup locations' mean fare
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[-20:])]['fare_per_mile_per_second'].mean()

Almost 10x the fare of least expensive locations

#### Number of trips in Expensive pickup locations
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[:20])].shape[0]

#### Number of trips in Inexpensive pickup locations
df.loc[df['PULocation'].isin(sorted_pickup_locations.keys()[-20:])].shape[0]

Close to 5x the number of trips in inexpensive areas

#### Expensive dropoff locations' mean fare
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[:20])]['fare_per_mile_per_second'].mean()

#### Inexpensive dropoff locations' mean fare
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[-20:])]['fare_per_mile_per_second'].mean()

Almost 10x the fare of least expensive locations

#### Number of trips in Expensive dropoff locations
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[:20])].shape[0]

#### Number of trips in Inexpensive dropoff locations
df.loc[df['DOLocation'].isin(sorted_dropoff_locations.keys()[-20:])].shape[0]

More than 2x the number of trips in inexpensive areas

### Exploratory Data Analysis (EDA) - 10
- Plot the average fare for popular pickup and dropoff locations
- Most of these locations are Airports, Tourist destinations
- Among these locations, some locations have average fare of \\$0.01 per mile per second whereas some have \\$0.04 per mile per second (almost 4x times the least one)
- Popular dropoff locations also follow similar pattern with fare ranging from 1x-6x

### Data Cleaning/Processing - 11
- In order to understand the true cost of each trip and the driver profits after deducting their gas cost, new columns like "driver_expenditure", "driver_profit" and "taxi_company_profit" are created.
- This is used in later phases to predict the base fare and how little or more the taxi company can charge the passenger and still end up in profit.
- Average mileage and gas price for the month are hyperparameters. We can choose the value according to the car type and gas prices

### Exploratory Data Analysis (EDA) - 11
- Plotting the "Driver's Profit" after deducting the expenses such as gas
- Plotting "Taxi Companies' Profit" after removing Driver's share from the Base passenger fare
- Driver's mostly get paid \\$5-\\$8 per trip with more pay for longer trips
- Taxi Companies profit is within a range for most trips with the range being \\$1-\\$5 (more pay for longer trips)

## Conclusion
- All the findings for each EDA are listed above, near each cell
- Trip time, trip miles, pickup location, dropoff location, driver's profit, taxi company profit margin effect the base passenger fare
- Other factors like tips also effect final price
- All of these factors will be taken into consideration when developing the Machine Learning models in Phase 2

## References
- https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- https://registry.opendata.aws/nyc-tlc-trip-records-pds/
- Seaborn, Pandas, Numpy documentation
