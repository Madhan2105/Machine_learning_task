import pandas as pd
import numpy as np
dataset = pd.read_excel('Sample Data (1).xlsx') 
print(dataset)

size = 48964
min_date = pd.to_datetime('2019-01-01')
max_date = pd.to_datetime('2020-12-12')
d = (max_date - min_date).days + 1
city = ['Mumbai', 'Pune', 'Other', 'Bengaluru']

df = pd.DataFrame() 
df['User ID'] = pd.RangeIndex(start=3262,stop=size+3262)
df['Session Count'] = np.random.uniform(low=1.0, high=500.0, size=(size,))
df['Uninstall Date'] = min_date + pd.to_timedelta(pd.np.random.randint(d,size=size), unit='d')
df['usertime'] = np.random.uniform(low=1.0, high=70608.384, size=(size,))
df['Share Count'] =   np.random.randint(0,200,size=len(df))
df['Notification Receive'] =   np.random.randint(0,1000,size=len(df))
df['Notification Dismiss'] =   np.random.randint(0,1000,size=len(df))
df['Notification Open'] =   np.random.randint(0,200,size=len(df))
df['Acquired Medium google / others'] = ["Normal"] * size 
df['Reg at date'] = min_date + pd.to_timedelta(pd.np.random.randint(d,size=size), unit='d')
df["city"] = np.random.choice(city, size=len(df))
df['Video Count'] =   np.random.randint(0,200,size=len(df))
df['Quiz Count'] =   np.random.randint(0,200,size=len(df))
# print(df.dtypes)
print(dataset)
print("--------------------------------")
print(df)
final_df = pd.concat([df,dataset])
print(final_df)
df.to_excel("Final_Expanded.xlsx",index=False)