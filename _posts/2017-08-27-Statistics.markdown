---
layout: post
title:  "Statistics 101"
date:   2017-08-27 20:30:15 +0100
categories: statistics, python
mathjax: true
---
"When you begin to know the intermediate/advanced, learn the basics again. They'll be useful."

Let's dive into some basic statistics! I will be using python to demonstrate topics using the pandas package and house price data obtained from <a href="https://www.ros.gov.uk/property-data/house-price-search">Registers of Scotland</A>. In the future I plan to revisit this page and rewrite for R.

Skip to...

[Mean, median and mode](#mean-median-and-mode)

[Variance and Standard Deviation](#variation-and-standard-deviation)

----
Before we get started I need to obtain my data, clean my data and import it. Frustratingly I can only obtain six months' worth of data at a time, and it is only displayed in the web browser in the form of a table. In the future writing a script that will read the data off the table and covert it into a csv file would be ideal, but for now manually copying and pasting the data into an excel spreadsheet and saving as a .csv file will have to do. I want to see how the house prices in Glasgow city centre (postcode G4 1) have changed throughout the 2000s and 2010s.

I save and then import into a pandas dataframe. 

{%highlight python%}
import pandas as pd
columns=['Date','Price']
g4_1 = pd.read_csv("data/G4_1__Housesales.csv", names=columns, index_col=None, encoding='latin-1')
g4_1 = g4_1.dropna()
{%endhighlight%}

To get my import to work correctly I needed to specify the encoding as latin-1. My data was also messy in that I have many empty lines and cells also being imported; these disappear when dropna() is called. Dropna() returns a new dataframe and doesn't update the existing one without being explicitly called to do so, as above.

Let's check how it's looking by calling **g4_1.head()** and previewing the first few entries.


|  | Date | Price |
|------|------|------|
| 0 | 15/03/2010 | £172,500.00 |
| 4 | 22/06/2010 | £165,000.00 |
| 8 | 23/03/2010 | £120,000.00 |
| 12 | 14/05/2010 | £31,500.00 |
| 15 | 17/06/2010 | £162,500.00 |

Looks alright to me! Those empty lines have disappeared and I'm left with an indexing of rows in multiples of 4. That is not so ideal for the perfectionist in me, but luckily this is easily ixed. 

{%highlight python%}
g4_1=g4_1.reset_index(drop=True)
g4_1.head()
{%endhighlight%}

 
|  | Date | Price |
|------|------|------|
| 0 | 15/03/2010 | £172,500.00 |
| 1 | 22/06/2010 | £165,000.00 |
| 2 | 23/03/2010 | £120,000.00 |
| 3 | 14/05/2010 | £31,500.00 |
| 4 | 17/06/2010 | £162,500.00 |

Much better! That drop=true argument ensures that the indexing replaces the previous indexing and isn't inserted as a new column. Next, both my Date and Price columns are in String format but for data analysis it is going to be an advantage to change these to more appropriate formats. Pandas, being the extremely convenient package it is, has handy functions that will transform these columns into the appropriate formats!

{%highlight python%}
g4_1['Date'] = pd.to_datetime(g4_1['Date'])
g4_1['Year of Sale'] = g4_1['Date'].dt.year
{%endhighlight%}

The pandas 'datetime' format is very smart and can handle most combinations of day/month/year/time formatting and convert it into a datetime object. I can then access individually the day, month, year etc of each data point. Here I have created a new column so that I can filter my dataset by year easily.

Finally I want to clean up the price column. I need to remove those pound signs and commas and convert the entire column of strings into a column of numbers.

{%highlight python%}
g4_1['Price'] = g4_1['Price'].str.replace(',','')
g4_1['Price'] = g4_1['Price'].str.replace('£','')
g4_1['Price'] = pd.to_numeric(g4_1['Price'])
{%endhighlight%}

These lines replace the £ and , with nothing (i.e. remove them) and converts the column of string into a column of numeric objects. Now pandas can understand the Price column as a column of numbers rather than strings. Here's what we have now:

 
|  | Date | Price | Year of Sale |
|------|------|------|------|
| 0 | 2010-03-15 | 172500.0 | 2010 |
| 1 | 2010-06-22 | 165000.0 | 2010 | 
| 2 | 2010-03-23 | 120000.0 | 2010 |
| 3 | 2010-05-14 | 31500.0 | 2010 |
| 4 | 2010-06-17 | 162500.0 | 2010 |


Good! Out of interest I'd like to plot a histogram of my dataset. I want there to be 70 bins in the range 0 to 1500000 and to count the number of occurences i.e. houses valued in each bin. This is very easily done.

{%highlight python%}
g4_1['Price'].plot.hist(by=None,bins=70, range=[0,1500000])
{%endhighlight%}

![270817__hist_fulldataset]({{site.baseurl}}/images/270817__hist_fulldataset.png){:class="img-responsive"}

What's important to take note here is that there are a few outliers beyond the £500k mark.

## Mean, Median and Mode ##
Often we want to look find the **average** value of a dataset; that is, where the middle of the data lies. Information on the nature of the data can often be gleamed from the average, but what information depends on which average you want to look at. There are three ways to measure the average: **mean**, **median** and the **mode**.

### Mean ###
The word average is commonly used in place of the mean. Take the sum of the samples and divide it by the number of samples and you have the mean. Very simple!

The _sample_ mean is given by $\bar{x}$ and with $n$ values in a dataset is defined as
\\[\bar{x} = \frac{1}{n} \displaystyle\sum_{i=1}^n x \\]

A distinction is made between the sample mean and the _population_ mean, which is calculated in the same way but is denoted as

\\[\mu = \frac{1}{n} \displaystyle\sum_{i=1}^n x \\]

The population mean is the average of the entire population and is usually impossible to compute. The mean is useful as it minimises error in the prediction of any one value in the dataset, and is the only measure of average where the sum of deviations of each value from the mean is always zero.

 So let's put it into practice, what is the mean Price of my entire dataset - that is, 17 years' worth of house price data in Central Glasgow?

{%highlight python%}
In: g4_1['Price'].mean()
Out: 126523.72705142858
{%endhighlight%}

So the mean value is £126,523.73... that's all well and good, but what does that actually mean? This value doesn't give us much information on its own, we need something to compare it to. What if we look at the mean for each year in our dataset? This is very possible with pandas.

{%highlight python%}
g4_1.groupby('Year of Sale')['Price'].mean().plot.bar()
{%endhighlight%}
 
![270817__groupby_mean]({{site.baseurl}}/images/270817__groupby_mean.png){:class="img-responsive"}

What you can immediately see here is that the trends aren't as smooth as you might expect - you can generally see a peak around 2006-2007 (gee, I wonder what happened then that sent everything downhill? ;)) but you also have 'anomalies' in 2004, 2009, 2011 and 2014 that makes the data seem noisier than it otherwise would might been.


### Median ###
The median is the middle number in a dataset when the dataset is in numerical order. If there are two numbers in the middle then the median is the mean of these two numbers. 

What's the median of my entire dataset?


{%highlight python%}
In: g4_1['Price'].median()
Out: 80000.0
{%endhighlight%}

That's a very different number to the mean - it's only 63% of the value! Why the discrepancy? Well the mean is very sensitive to abnormal values, wheras the median is not significantly changed by outliers. So having a handful of houses sold for a large amount will affect the mean but not so much the median. When you have a clean dataset with few outliers these values will be in agreement with each other but apparently my dataset is not in this category.


{%highlight python%}
g4_1.groupby('Year of Sale')['Price'].median().plot.bar()
{%endhighlight%}
 
![270817__groupby_median]({{site.baseurl}}/images/270817__groupby_median.png){:class="img-responsive"}

I would say the trends here are "smoother" i.e. year-on-year you see smoother liners and fewer abnormal years like you saw in the mean. From this I might conclude that the yearly median value would be a more useful datapoint to gleam information from than the mean.

### Mode ###

The mode is the value that appears most often in a dataset. 

{%highlight python%}
In: g4_1['Price'].mode()
Out: 
0	75000.0
dtype: float64
{%endhighlight%}

You might notice the output is different to that of the median and the mean. That is because the mode is not necessarily unique, and there may be multiple values for the mode in a dataset and so the output is in the form of an array - in this case, an array of length 1. Another problem with the mode is when we have continuous data we are more likely not to have any one value that is more frequent than the other. For example how likely are we to have more than one house that sold for exactly £253,400. Not very likely. We can see just how many houses in our dataset sold for £75,000 by calling value_counts(), which counts the number of unique values in a dataset.

{%highlight python%}
In: g4_1['Price'].value_counts()
Out: 
75000.0      37
80000.0      35
70000.0      28
             ..
208000.0      1
1600000.0     1
1000.0        1
Name: Price, Length: 663, dtype: int64
{%endhighlight%}



## Variation and Standard Deviation ##

The mean, median and mode do a good job telling us where the centre of a dataset is. But often we also want to measure how far the data is spread around some central value. 

### Variance ###

The *sample variance* in a distribution with *n* data points is defined as

\\[s^{2} = \frac{1}{n-1} \displaystyle\sum_{i=1}^n (x_i-\bar{x})^2 \\]

i.e. the average of the squared differences from the mean. The variance is a measure of how "spread-out" the data is.

The *population variance* is defined as

\\[s^{2} = \frac{1}{n} \displaystyle\sum_{i=1}^n (x_i-\mu)^2 \\]

where the denominator is *n* as opposed to *n-1*. While the *sample mean* is an unbiased estimator of the *population mean*, the sample varince is not of the population variance. 

### Standard Deviation ###

The *standard deviation* is the square root of the variance. It is a measure of dispersion of observations within a dataset. Being the square root of the sum of squared value the units of standard deviation are the same as those in the dataset. Handy!

That's all for now! Next time: Probability density functions!

