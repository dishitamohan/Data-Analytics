# IDA

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import scipy.stats as stats
import datetime 
df1=pd.read_csv('Cereals.csv')



1.	iloc[row:row,col:col]

1.	df.describe() //5 number summary

2.	typevsmfr=pd.crosstab(df1.type, df1.mfr)

3.	print("The creal with best rating: ")
 maxm=df1['rating'].idxmax()
print(df1.iat[maxm,0])
print(df1['rating'].max())

4.	correlation= df1['calories'].corr(df1['rating'])

5.	mfr1=df1[df1['mfr']=='A']

6.	mfr1['rating'].mean()

7.	rating=pd.DataFrame({'mfr':['A','G','K','N','P','Q','R'],
'Rating':[avg1,avg2,avg3,avg4,avg5,avg6,avg7]})

8.	df.skew()
9.	df.kurt())
10.	df.shape[0]  //0 no of rows(no of records)
11.	df['Genre'].str.strip() //trim
12.	gen.value_counts(sort=True, ascending=False)
13.	df['month']=pd.DatetimeIndex(df['Release Date']).month
14.	df['ROI']=(df['BoxOfficeCollection']-df['Budget'])/df['Budget']
15.	sum1=gen1['YoutubeLikes'].sum()
16.	yr1= year1['SlNo'].nunique()
17.	match_detail=df[['id', 'season']].merge(DF, left_on='id', right_on='match_id', how='left')

18.	match_detail.groupby(['season'])['total_runs'].sum().reset_index()

19.	top10.sort_values(by=['total_runs'], ascending=False).head(10)

20.	total = totalbat.append(totalbowl, ignore_index=True)
totalt= total.value_counts()
team_stats=pd.DataFrame({'TotalMatches': df.team1.value_counts()+ df.team2.value_counts()})
21.	team_stats=team_stats.reset_index()
22.	team_stats.rename(columns={'index': 'Teams'}, inplace=True)
23.	#Finding unique CustomerIDs and storing them in an array
CustomerID = OR3['CustomerID'].unique()
CustomerID

RFM = pd.DataFrame(columns = ['CustomerID','Recency', 'Frequency','Monetary Value'])
RFM['CustomerID'] = CustomerID
RFM

RFM.sort_values(by ='CustomerID',inplace = True)

for CID in CustomerID:
    RFM.loc[RFM.CustomerID==CID, 'Frequency'] = OR3[OR3['CustomerID']==CID]['InvoiceNo'].value_counts().sum()
    RFM.loc[RFM.CustomerID==CID, 'Recency'] = (12-OR3.loc[OR3.CustomerID==CID,'InvoiceDate'].max().date().month)

    InvoiceArr=OR3.loc[OR3.CustomerID==CID,'InvoiceNo'].unique()
    sum=0
    for INo in InvoiceArr:
        sum+=((OR3.loc[OR3.InvoiceNo==INo,'UnitPrice']*OR3.loc[OR3.InvoiceNo==INo,'Quantity']).sum())
    RFM.loc[RFM.CustomerID==CID, 'Monetary Value'] = sum
RFM


RFMc = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary Value'])
RFMc['Recency'] = RFM['Recency'].dropna()
RFMc['Frequency'] = RFM['Frequency'].dropna()
RFMc['Monetary Value'] = RFM['Monetary Value'].dropna()
RFMc

24.	df = pd.read_excel('employment.xlsx', parse_dates=['datestamp'], index_col='datestamp')

25.	df.set_index('datestamp')

26.	
 
# VISUALIZATION

1.	sns.boxplot(x='rating' ,y='type', data=df1)
plt.title("side-by-side boxplot comparing the consumer rating of hot vs. cold cereals.")
plt.show()

2.	sns.pairplot(df[['sugars','calories','carbo','fat']])
plt.show()

3.	plt.plot(rating['mfr'], rating['Rating'])
plt.title("relation between manufacturer and rating")
plt.show()
//line plot.     //cat vs num
4.	plt.hist(df['mpg'])
plt.title("miles per gallon")
plt.xlabel("Miles")
plt.ylabel("Gallons")
plt.show()

//distribution.

5.	sns.heatmap(df.corr())
plt.show().    //correlation

6.	df3[['Genre', 'Budget']].plot(kind='bar')
plt.show  //distribution. // categories not printed only index
7.	ax = plt.axes()
ax.set(facecolor = "white")
sns.countplot(x='season', hue='toss_decision', data=df,palette="gnuplot2",saturation=1)

plt.xlabel('\n Season',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('Toss decision in different seasons',fontsize=15,fontweight="bold")
plt.show()

// Visualize the Toss decision across seasons

8.	sns.barplot(x = top_players.index, y = top_players, orient='v'); #palette="Blues");
plt.show()
9.	sns.lineplot(data=df.iloc[:,3:6])

sns.set(rc={"figure.figsize":(15,15)})
sns.set(font_scale=2)
plt.show() 

//vertical lines one over another //distribution

10.	sns.lineplot(data=df.iloc[:,2:4])
sns.set(rc={"figure.figsize":(15,15)})
sns.set(font_scale=2)
plt.show()

//horizontal lines one over another //distribution

11.	#Scatterplot: Frequency vs Recency
facet = sns.lmplot(data=RFMc, x='Recency', y='Frequency', hue='Cluster', 
                   fit_reg=False, legend=True, legend_out=True)
plt.legend(loc='right', labels=['Occasional Customers(0)', 'Routine Customers(1)' ,'VIP Customers(2)'])


12.	#Visualising clusters usind dendrogram

Dendrogram = shc.dendrogram((shc.linkage(Xac, method ='ward')))

13.	# 2. Generate a boxplot to find the distribution of unemployment rate for every industry.

sns.boxplot(data=df)
plt.title("distribution of unemployment rate for every industry")
sns.set(rc={"figure.figsize":(15,15)})
sns.set(font_scale=2)
plt.show()

14.	# 4. Plot the monthly and yearly trends.
while (i<=17):
    sns.lineplot(x='month',y=df.iloc[:, i],data=df)
    i=i+1
    
plt.xlabel("Month")
plt.ylabel("Rate")
plt.legend(labels=df.iloc[:, 1:17], title = "Title_Legend")




















# DATA CLEANING
1.	df.replace(item, replace) //data cleaning
2.	np.nan //fill up nan values
3.	df['potass']=pd.to_numeric(df['potass'], errors='coerce') //if 'coerce', then invalid parsing will be set as NaN
4.	df=df.mask(df==-1)  //The mask() method replaces the values of the rows where the condition evaluates to True
5.	df= df.fillna(df.mean())
6.	q1=df.quantile( 0.25)
iqr=q3-q1
low_lim = q1 - 1.5 * iqr
df=df[df>low_lim]
df= df.fillna(df.median())
7.	df=df.dropna()  //dropping empty fields
8.	# 4. Clean the data and remove the special characters and replace the contractions with its expansion by converting the uppercase character to lower case. Also, remove the punctuations.

Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
           "'d":" would","'ve":" have","'re":" are"}
 
#replace the contractions
for key,value in Apos_dict.items():
    if key in df['review']:
        df['review']=df['review'].replace(key,value)

9.	# removing non alpha-numeric characters
df['review'] = df['review'].str.replace('[^a-zA-Z0-9]', ' ', regex=True).str.strip()
10.	#converting to lower case
df['review'].str.lower()

11.	# 5. Add the Polarity, length of the review, the word count and average word length of each review


df = df[df['rating'] != 3]
def Polarity(n):
    return 1 if n >= 4 else 0
df['Polarity'] = df['rating'].apply(Polarity)

df['Length_of_review'] = df['review'].apply(len)

df['Word_count'] = df['review'].str.count(' ')+1

df['avg_word_length'] = df['Length_of_review']/ df['Word_count']

12.	OR3 = OR2.loc[OR2.Quantity>0]


#AMAZON BABY

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from html.parser import HTMLParser
import re   
# APRIORI


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from csv import reader
from mlxtend.plotting import plot_decision_regions

import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

1.	a= []
with open('groceries.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        a.append(row)
2.	b= TransactionEncoder()
trans= b.fit(groceries).transform(a)
trans

3.	trans= transactions.astype('int')
trans

4.	df = pd.DataFrame(trans, columns=encoder.columns_)
5.	freq= apriori(df, min_support=0.02, use_colnames=True)
freq['length'] = freq['itemsets'].apply(lambda x: len(x))
freq

6.	print("TOP 10 SELLING ITEMS ON THE BASIS OF SUPPORT(2%): ")
freq = freq.sort_values(by='support', ascending=False)
freq[ (freq['length'] == 1) & (freq['support'] >= 0.02) ][0:10]

7.	x= association_rules(freq, metric='support', min_threshold=0.02)
print("DATAFRAME ACC.TO CONFIDENCE LEVEL IS AS FOLLOWS: ")

x.sort_values(by='confidence', ascending=False)[0:10]

8.	print("DATAFRAME ACC.TO LIFT CONDITION IS: ")
x[(x['support'] >= 0.02) & (x['lift'] > 1.0)]
 
# CORRELATION

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import datetime

from sklearn.metrics import pairwise_distances as pair
from scipy.spatial.distance import cosine, correlation

1.	#2. Read the “ratings.csv” file and create a pivot table with index=‘userId’, columns=‘movieId’, values = “rating".

dff = df3.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0, aggfunc=np.mean)

2.	#Use cosine similarity for finding similarity among users. Use the following packages

pair=pairwise_distances(dff, Y=None, metric='cosine')

3.	#6. Find the 5 most similar user for user with user Id 25.

    print((-pair[24]).argsort()[1:6]+1)

4.	#7. Use the “movies” dataset to find out the names of movies, user 1 and user 338 have watched in common and how they have rated each one of them.
common_movies = set(df3.loc[df3.userId==1, 'movieId']).intersection(set(df3.loc[df3.userId==338, 'movieId']))

5.	#8. Use the movies dataset to find out the common movie names between user 2 and user 338 with least rating of 4.0

common_movies2 = set(df3.loc[((df3.userId==2) & (df3.rating>=4.0)), 'movieId']).intersection(set(df3.loc[((df3.userId==338) & (df3.rating>=4.0)), 'movieId']))

common_movies2


for movieid in sorted(common_movies2):
    common = df2[df2['movieId'] == movieid]
    print(common)

6.	#9. Create a pivot table for representing the similarity among movies using correlation.

dff = pd.pivot_table(df3, index='movieId', columns='userId', values='rating', fill_value=0)

pair2=pairwise_distances(dff, Y=None, metric='correlation')


7.	#10.Find the top 5 movies which are similar to the movie “Godfather”.

for movie in df2.title:
    if('Godfather' in str(movie)):
        print(movie)

df2.loc[(df2.title=='Godfather, The (1972)'), 'movieId'].values[0]

similar_movies = pair2[857].argsort()[1:6]+1

for movieID in similar_movies:
    print(df2[df2["movieId"] == movieID])
    print("\n")

 
# CLUSTERING

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
%matplotlib inline

from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as shc



1.	X = StandardScaler().fit_transform(RFMc)
2.	model = KMeans()
visualizer = KElbowVisualizer(model,k=(1,9))
visualizer.fit(X)
visualizer.show()

3.	k_means = KMeans(n_clusters=3)
model = k_means.fit(X)
model
4.	RFMc['Cluster'] = k_means.predict(X)

5.	RFMc0 = RFMc[RFMc['Cluster']==0]
for col in ['Recency', 'Frequency', 'Monetary Value']:
    print(col)
    print("Min: ", RFMc0.loc[:,col].min())
    print("Median: ", RFMc0.loc[:,col].median())
    print("Mean: ", RFMc0.loc[:,col].mean())
    print("Max: ", RFMc0.loc[:,col].max())
    print("")
print("Total Monetary Value:", RFMc0['Monetary Value'].sum())


6.	Xac = StandardScaler().fit_transform(RFMac)
7.	ac = AgglomerativeClustering(n_clusters = 3)
RFMac['Cluster'] = ac.fit_predict(Xac)



In these 3 scatterplots, it is seen that the cluster boundaries are much more clearer. They do not mix up with other clusters and have greater intra-cluster similarity Hence Agglomerative Clustering yeilds better cluster predictions in this case
