# importing pandas package 
import os 
import pandas as pd

# making data frame from csv file 
data = pd.read_csv("Sheet 3_usage_time_0_28.csv") 

# generating one row 
rows = data.sample(frac =.50) 

# checking if sample is 0.25 times data or not 

if (0.5 *(len(data))== len(rows)): 
	print( "Cool") 
	print(len(data), len(rows)) 

# display 
print(rows)
