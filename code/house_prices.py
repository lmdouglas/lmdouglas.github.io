#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:16:08 2017

@author: Lauren
"""

import pandas as pd

"""
filepath = "data/gov__house_prices__pp-2017.csv"
df = pd.read_csv(filepath, names=["Transaction ID","Price","Date of Transfer","Postcode","Property Type","Old/New","Duration","Primary Addressable Property Name","SAON","Street","Locality","Town/City","District","County","PPD Category Type","Record Status"])

df.plot.hist("Price",bins=60, range=[0,1500000])

print("Price mean = " + str(df["Price"].mean()))
print("Price median = " + str(df["Price"].median()))
print("Price mode = " + str(df["Price"].mode()))

df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"])
df["Month of Transfer"] = df["Date of Transfer"].dt.month

#Take average of each month
df.groupby("Month of Transfer")["Price"].median().plot.bar()

#Take average of each postcode (first letter)
df["Postcode_Letter"] = df["Postcode"].astype(str).str[:1]

median_price_by_postcode = df.groupby("Postcode_Letter")["Price"].median()
mean_price_by_postcode = df.groupby("Postcode_Letter")["Price"].mean()

#plt = df.groupby("Postcode_Letter")["Price"].median().plot.bar()
#plt.set_ylim(500000,2000000)

#Let's have a look at Glasgow...
#Postcode G41 specifically
#df_g41 = df[df["Postcode"].str.contains("G41", na=False)]
"""
"""
#Postcode: G41 1
g411 = pd.read_csv("data/G41_1__2016_and_2010_Housesales.csv", index_col=None, encoding='utf-8-sig')
g411=g411.dropna()
g411['Date'] = pd.to_datetime(g411['Date'])
#Set date as the index of rows
#Can then select rows by date using df.loc[start_date:end_date]
#g411=g411.set_index(['Date'])

g411['Year of sale'] = g411['Date'].dt.year

g411['Price'] = g411['Price'].str.replace(',','')
g411['Price'] = g411['Price'].str.replace('£','')
g411['Price'] = pd.to_numeric(g411['Price'])

g411.groupby('Year of sale')['Price'].median().plot.bar()
g411.groupby('Year of sale')['Price'].mean().plot.bar()

#Look at just 2011 data
g411_2011 = g411[(g411['Date'] > '2011-01-01 00:00:00') & (g411['Date'] < '2011-12-31 00:00:00')]
g411_2011['Price'].plot.hist(by=None,bins=45, range=[0,1500000])

print("Median = " + str(g411['Price'].median()))
print("Mean = " + str(g411['Price'].mean()))

"""

#Looking at postcode G4 1"
g4_1 = pd.read_csv("data/G4_1__2016_and_2010_Housesales.csv", index_col=False, encoding='latin-1')
g4_1 = g4_1.dropna()

print(g4_1.head())

#Convert date column to datetime
g4_1['Date'] = pd.to_datetime(g4_1['Date'])
g4_1['Year of Sale'] = g4_1['Date'].dt.year

#Clean up price column
g4_1['Price'] = g4_1['Price'].str.replace(',','')
g4_1['Price'] = g4_1['Price'].str.replace('£','')
g4_1['Price'] = pd.to_numeric(g4_1['Price'])


g4_1.groupby('Year of Sale')['Price'].mean().plot.bar()
g4_1.groupby('Year of Sale')['Price'].median().plot.bar()

g4_1['Price'].plot.hist(by=None,bins=20)