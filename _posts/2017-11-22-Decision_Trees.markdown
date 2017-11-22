---
layout: post
title:  "Decision Trees"
date:   2017-11-22 11:31:15 +0100
categories: machinelearning decisiontrees
---

### Decision Trees ###

In *decision analysis*, decision trees are decision support tools that use a tree-like graph or model of decisions and possible consequences. It is an algorithm containing only conditional (if... then...) control statements.

![decisiontree_swengorphys]({{site.baseurl}}/images/decisiontree_swengorphys.jpeg){:class="img-responsive"}

A decision tree gives you a flowchart of making decisions. For example, say I inherited a piece of code and I would like to determine whether the code was originally written by a physicist or by a software engineer, just by looking at certain characteristics for the code. Here, the author is a dependent variable, and there are various aspects of the code that might influence the final decision. (This is all in jest here of course, mostly...) 

Decision trees are some of the most intuitive classifiers as they are easy to construct and to understand. Internal 'nodes' correspond to attributes (or features), the leafs from these nodes correspond to classification outcome, and the edges denote final assignment. 

Decision trees are a form of *supervised learning*. This means that a training dataset must be provided, with inputs and known/desired outputs, that the decision tree can learn from. The optimal scenario allows for the algorithm to correctly determine the class labels for *unseen* data. 


### How decision trees work ###

At each step, we want to find the attribute we can use to partition the dataset in order to minimise the *entropy* of the data at the next step. 

If you're from a similar background to me, you may understand entropy in a thermodynamical sense, i.e. it is a measure of (dis)order of a system. A higher entropy corresponds to a greater degree of disorder. Essentially this is what we mean here, where entropy corresponds to the amount of *uncertainty* associated with a specific probability distribution. So the higher the entropy, the less confident we are in the outcome. So if we minimise the entropy at each step we are making a decision at each step with the highest confidence. 

At some point in the near future I hope to go through the process of coding my own decision tree algorithm and go through in detail how it works, but for now let's just explore how we might make a decision tree in python using numpy, scikit-learn and pandas.

### Decision trees in python ###

![pandas]({{site.baseurl}}/images/pandas.jpg){:class="img-responsive"}

I love using *pandas* dataframes, they are powerful and flexible. Before pandas it never made sense to use python over R or SQL for data analysis, but now python is a great contender and is increasingly becoming one of the most popular choices of language for data scientists.  

Something that is a rite of passage for aspiring data scientists is exploring the [kaggle titanic dataset](https://www.kaggle.com/c/titanic) which we will be using today. A training dataset, a subset of the total passenger list with survival outcomes, is provided to train whichever algorithm you are running. The idea is then to predict passenger survival based on given attributes in the test dataset. 

Let's load up the test dataset in a pandas dataframe and have a look at it.

{%highlight python%}
import numpy as np
import pandas as pd
from sklearn import tree

input_file = "train.csv"
ds = pd.read_csv(input_file, header=0)

df.head()
{%endhighlight%}

| PassengerId | Survival | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked
|------|------|------|
| 1 | 0 | 3 | Braud, Mr. Owen Harris| male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S
| 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th..| female | 38.0 | 0 | 0 | PC 17599 | 71.2833 | C85 | C
| 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | ST0N/02. 3101282 | 7.9250 | NaN | S
| 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath ( Lily May Peel ) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S
| 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S

What some of these columns represent isn't immediately obvious and so we need to refer to the [data guide](https://www.kaggle.com/c/titanic/data).

*Pclass*: Socio-economic class of passenger; 1st, 2nd or 3rd.

*age*: Fractional if less than 1, in the form of xx.5 if estimated,

*sibsp*: Number of siblings or spouses aboard.
Sibling = brother, sister, stepbrother, stepsister.
Spouse = husband or wife.

*parch*: Number of parents/child aboard.
Parent = mother, father.
Child = daughter, son, stepdaughter, stepson.
parch=0 if child travelled with a nanny. 

Well that's cleared that up a little bit. 

Scikit-learn needs everything to be numerical in order for decision trees to work, so we will need to transform some of these columns. As a very first approach, simply in order to get things going, I will disregard the Name, Cabin and the Ticket columns for now, as these require a little more feature engineering to gleam useful information out of. All the rest of our columns are in numerical form except for Sex and Embarked, which describes whether the passenger embarked on the titanic at *C*herbourg, *Q*ueenstown or *S*outhampton. We can transform this data by *mapping* it to numbers.

{%highlight python%}
embarked = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked'] = df['Embarked'].map(embarked)
sex = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex)
{%endhighlight%}

Next we need to identify the features we wish to build the tree on from the target column we are trying to build the tree for ('Survival').

{%highlight python%}
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
{%endhighlight%}

And construct the decision tree:
{%highlight python%}
target = df['Survival']
dt_features = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(dt_features, target)
{%endhighlight%}

However when we try and run this, we get the following error:

{%highlight python%}
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
{%endhighlight%}

So it appears that some of the data is missing and is imported into pandas as NaN; Not a Number. Scikit-learn's tree algorithm can't cope with this value. So what to do?

As far as I can tell this is a hotly contested issue... if there are only one or two entries with NaN in any one column, perhaps easiest is to skip it? Let's count the occurences in each column:

{%highlight python%}
for feature in features:
    print(feature,"Null:",df[feature].isnull().values.sum())
{%endhighlight%}

{%highlight python%}
Pclass Null: 0
Sex Null: 0
Age Null: 177
SibSp Null: 0
Parch Null: 0
Fare Null: 0
Embarked Null: 2
{%endhighlight%}

So maybe we could choose to ignore those 2 entries with NaN embarked values, but we would really like to not lose those 177 entries with a missing value for Age! One strategy we can employ here is to replace those missing values with the median values for that dataset. 

{%highlight python%}
median_age = df['Age'].median()
df['Age'] = df['Age'].replace(np.NaN, median_age)

median_age = df['Embarked'].median()
df['Embarked'] = df['Embarked'].replace(np.NaN, median_age)
{%endhighlight%}

I have chosen the median here in case our data has many outliers as the median is not as strongly affected by them.

Now let's try again with our decision tree... and display it

{%highlight python%}
from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
{%endhighlight%}

This code exports the decision tree in DOT format, which is a GraphViz representation of the decistion tree. From this DOT format graphic renderings can be generating, called by the create_png() function. The resulting decision tree is huge, I attach here merely a portion of it!

![decisiontree_titanic_small]({{site.baseurl}}/images/decisiontree_titanic_small.png){:class="img-responsive"}
 
[Information on reading this]

Now that we have a decision tree, let's make some predictions! I want to read in the test.csv data first of all, make the predictions and then save the predictions to a file ready for scoring by Kaggle.

As I need to set up the test data identically to the training data, I compile the relevant lines above into a function called setup_data.

{%highlight python%}
test_file = "test.csv"
test_df = pd.read_csv(test_file, header=0)
setup_data(test_df, features) #Map data, replaces NaNs with medians
test_df_features = test_df[features] #Create a new dataframe just of columns listed in "features"
predictions = clf.predict(test_df_features) #Use decision tree to predict survival rates
{%endhighlight%}

Done! Now we need to save the data in a kaggle-friendly format, i.e. in a csv file with two columns, LHS with PassengerID, RHS with survival predictions.

{%highlight python%}
df_predictions = pd.DataFrame({'PassengerID': test_df['PassengerId'], 'Survived': predictions}) #Create dataframe, one column of PassengerIDs, one of Survival prediction
df_predictions.to_csv(path_or_buf="predictions_3.csv", index=False) #Save pandas dataframe to csv file
{%endhighlight%}

And get it marked by Kaggle...

![kaggle_submission_1]({{site.baseurl}}/images/kaggle_submission_1.png){:class="img-responsive"}

So a simple decision tree against our training dataset leads to a prediciton accuracy of **67.9%** in the test dataset. So not bad, a majority of it is correct, but there is definite room for improvement! Considering an extremely simple sex-based model is around 70% accurate, this model so far leaves a lot to be desired.

Decision trees are prone to *overfitting*, which means that they are too accurate on the training data and therefore fails to predict future data.

### Random Forests ###

Something we can employ here to help improve the model is a *random forest*. The idea is to train using multiple decision trees. Each tree gets a random sample, and each resulting tree can then "vote" on the correct result. 

Random forests are an *ensemble learning* method, which is a method where multiple learning algorithms are used to obtain better predictive performance. 

Random forests are fairly easy to construct.

{%highlight python%}
clf_rf = RandomForestClassifier(n_estimators=10)
clf_rf = clf_rf.fit(dt_features,target)
predictions = clf_rf.predict(test_df)
{%endhighlight%}

And upload to kaggle...

![kaggle_submission_2]({{site.baseurl}}/images/kaggle_submission_2.png){:class="img-responsive"}

So our random forest method has improved the accuracy of the model to **72.7%**. So I probably wouldn't use this model to choose where to invest my many millions of pounds (ha!) but at least we are seeing improvements here!

In the future I will revisit this set and try out some other machine learning techniques, and also will look into how to use the non-numerical information in this dataset that I've ignored this time round!


{%comment%}



At each step, find the attribute we can use to partition the dataset ot minimise the entropy of the data at the next step. i.e. we have a resulting set of classifications and we want to choose the attribute decision that will minimise the entropy at the next step e.g. at each step we want to make all of the remaining choices result in either as many more hires or as many no-hires as possible i.e. make it more uniform.
Algorithm: ID3
Greedy algorithm - picks attribute that minimises entropy at that point - might NOT be optimal and minimises the number of choices that you have to make. But will work.

Prone to overfitting! 

Use **random forests**
Idea: sample data that we train on in different ways from multiple decision trees. each tree gets random sample and constructs tree. each resulting tree can vote on the right result. 
boot strap aggregating - bagging - randomly re-sampling input data for each tree. ensemble learning.
helps combat overfitting.
Can restrict the number of attributes at each stage that a tree can choose from - gives more variation from tree to tree. 

{%endcomment%}


