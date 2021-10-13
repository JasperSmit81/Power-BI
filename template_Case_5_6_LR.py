#!/usr/bin/env python
# coding: utf-8

# # Project Title 

# In[1]:


project_title="Case_week_5_6"


# # 1 system setup

# 
# **Credits :**   
# https://github.com/Kaushik-Varma/Marketing_Data_Analysis   
# https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14   
# https://seaborn.pydata.org/generated/seaborn.boxplot.html   
# 

# **Working directory setup**
# * **/Data/** for all data related maps
# * **/Data/raw/** for all raw incoming data
# * **/Data/clean/** for all clean data to be used during analysis
# * **/Data/staging/** for all data save during cleaning 
# * **/Data/temp/** for all tempral data saving 
# * **/Figs/temp/** for all tempral data saving 
# * **/Docs/** reference documentation
# * **/Results/** reference documentation
# * **/Code/** reference documentation
# 
# 
# **references:**
# https://docs.python-guide.org/writing/structure/
# 
# 

# Setup packages required for analysis
# 

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import seaborn as sns
import plotly.express as px
import requests
import plotly.graph_objects as go
import json
import folium
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import datetime as dt
from matplotlib.widgets import CheckButtons
from matplotlib.pyplot import figure
from jupyter_dash import JupyterDash
import dash
import dash_html_components as html
import dash_core_components as dcc
from powerbiclient import Report, models
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option('display.max_columns', None)


# Set working directories 

# In[4]:


print("Current working directory: {0}".format(os.getcwd()))


# In[5]:


working_dir = os.getcwd()


# Code below created project structure

# In[6]:


arr_map_structure  = [os.getcwd() + map for map in   ['/Data','/Data/raw','/Data/clean','/Data/staging',
                      '/Data/temp','/Figs','/Figs/temp','/Docs','/Results','/Code'] ]

[os.makedirs(map) for map in arr_map_structure if  not os.path.exists( map)]


# In[7]:


raw_data_dir = working_dir +'/Data/raw/'


# # 2 Import data

# show contents of working directory

# In[8]:


os.listdir(raw_data_dir)


# In[9]:


#inladen van de api's en opslaan als aparte dataframes
data_lpg = pd.read_csv('laadpaaldata.csv')
url_ocm = 'https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=100&compact=true&verbose=false&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017%22)'
url_rdw = 'https://opendata.rdw.nl/resource/8ys7-d773.json'


data_ocm = json.loads(requests.get(url_ocm).content)
df_ocm = pd.json_normalize(data_ocm)

data_rdw = json.loads(requests.get(url_rdw).content)
df_rdw = pd.json_normalize(data_rdw)


# In[10]:


df_rdw.head()


# In[11]:


data_lpg.head()


# In[12]:


df_ocm


# # 3 Exploratory Data Analysis

# In[13]:


whos


# ## 3.1 EDA per dataframe

# Eerst kijken naar de veldnamen en data types

# In[14]:


data_lpg.shape


# In[15]:


df_ocm.shape


# In[16]:


df_rdw.shape


# In[17]:


data_lpg.dtypes


# In[18]:


df_ocm.dtypes


# In[19]:


df_rdw.dtypes


# In[ ]:





# Dan kijken naar de statistieken van alle kolommen  

# In[20]:


df_rdw['brandstof_omschrijving'].value_counts()


# In[21]:


data_lpg.nunique(axis=0)


# In[22]:


df_ocm.columns


# In[23]:


df_ocm.isna().sum()


# In[24]:


df_rdw.nunique(axis=0)


# ## 3.2.1 kolommen splitsen / verwijderen / hernoemen / 

# **dataset van openchargemap schoonmaken**

# AdresInfo informatie zoals contactgegevens wordt verwijderd. AdresInfo zoals straatnaam en provincie wordt behouden omdat dit in een map duidelijker maakt waar het puntje op de map ligt

# In[25]:


df_ocm.head()


# In[26]:


#verwijderen van onnodige kolommen
df_ocm = df_ocm.drop(['AddressInfo.RelatedURL', 'AddressInfo.ContactTelephone1', 'AddressInfo.AccessComments', 'AddressInfo.AddressLine2', 'AddressInfo.ContactEmail', 'AddressInfo.ID', 'AddressInfo.CountryID', 'AddressInfo.DistanceUnit','DataProviderID','DataQualityLevel'],axis=1)


# In[27]:


df_ocm


# De tijd is op drie manieren weergegeven in deze dataset. Ten eerste de laatste verificatie data, die in dit geval niet veel uitkomst bieden want van slechts 5 rijen is deze waarde bekent. Verder zijn er twee kolommen met tijd informatie, de laatste keer dat de informatie die in de rij staat geupdate is en wanneer deze aangemaakt is.
# 
# De de verifictaie data wordt buiten beschouwing gelaten omdat de dataset anders te klein wordt. De tijd data in de andere twee kolommen wordt de datum gescheiden van de tijd en de tijd verwijdert omdat die informatie iets te specifiek is voor deze analyse

# In[28]:


type(df_ocm['DateLastStatusUpdate'])


# In[29]:


#Extract date  & time in newly from "DateLastStatusUpdate" column.
df_ocm['date_LSU']= df_ocm["DateLastStatusUpdate"].apply(lambda x: x.split("T")[0])
df_ocm['time_LSU']= df_ocm["DateLastStatusUpdate"].apply(lambda x: x.split("T")[1])


# In[30]:


# Drop the "jobedu" column from the dataframe.
df_ocm.drop('DateLastStatusUpdate', axis = 1, inplace = True)
df_ocm.drop('time_LSU', axis = 1, inplace = True)


# In[31]:


#Extract date  & time in newly from "DateLastStatusUpdate" column.
df_ocm['date_created']= df_ocm["DateCreated"].apply(lambda x: x.split("T")[0])
df_ocm['time_created']= df_ocm["DateCreated"].apply(lambda x: x.split("T")[1])


# In[32]:


# Drop the "jobedu" column from the dataframe.
df_ocm.drop('DateCreated', axis = 1, inplace = True)
df_ocm.drop('time_created', axis = 1, inplace = True)


# In[33]:


df_ocm = df_ocm.drop(['IsRecentlyVerified','DateLastVerified'],axis=1)


# In[34]:


df_ocm


# **dataset van rdw schoonmaken**

# Er wordt in deze analyse niet gekeken naar geluidsniveau, uitstoot of de mileuklasse. Er wordt hier gekeken naar de soorten brandstoffen en hoe deze gebruikt worden

# In[35]:


df_rdw.nunique(axis=0)


# In[36]:


df_rdw = df_rdw.drop(['co2_uitstoot_gecombineerd', 'geluidsniveau_rijdend', 'geluidsniveau_stationair', 'emissiecode_omschrijving', 'milieuklasse_eg_goedkeuring_zwaar', 'roetuitstoot', 'toerental_geluidsniveau', 'milieuklasse_eg_goedkeuring_licht', 'uitstoot_deeltjes_licht', 'emissie_co2_gecombineerd_wltp', 'co2_uitstoot_gewogen', 'emis_deeltjes_type1_wltp', 'emis_co2_gewogen_gecombineerd_wltp', 'uitstoot_deeltjes_zwaar'],axis=1).sort_values(by=['brandstof_omschrijving']).reset_index()


# In[37]:


df_rdw.isnull().sum()


# In een dataset met 1000 rijen als er dan in een kolom meer dan 900 (90%) NaN staan worden deze verwijderd, deze data is dan niet representabel genoeg voor de gehele dataset

# In[38]:


df_rdw = df_rdw.drop(['brandstof_verbruik_gecombineerd_wltp', 'klasse_hybride_elektrisch_voertuig', 'elektrisch_verbruik_enkel_elektrisch_wltp', 'actie_radius_enkel_elektrisch_wltp', 'actie_radius_enkel_elektrisch_stad_wltp', 'max_vermogen_60_minuten', 'max_vermogen_15_minuten', 'elektrisch_verbruik_extern_opladen_wltp', 'actie_radius_extern_opladen_wltp', 'actie_radius_extern_opladen_stad_wltp', 'brandstof_verbruik_gewogen_gecombineerd_wltp'],axis=1)


# In[39]:


df_rdw


# ## 3.2.2 Duplicates zoeken/verwijderen

# In[40]:


df_rdw.duplicated(keep='last').sum()


# In[ ]:





# ## 3.2.3 generieke verkenning

# In[41]:


df_ocm.dtypes


# In[42]:


df_rdw.dtypes


# Wat valt op:
# * df_ocm reageert niet op een aantal dataframe commands, misschien komt dit door de longitude en lattitude die er nog instaat misschien dat het omzetten naar een geojson/geodataframe helpt

# Welke relaties willen we toetsen
# * aantal voertuigen per maand per brandstof categorie
# * De laadtijd
# * waar in het land laadpunten zijn

# In[43]:


plt.scatter(x=df_ocm['AddressInfo.Longitude'],y=df_ocm['AddressInfo.Latitude'])
plt.show()


# In[44]:


gdf_ocm = gpd.GeoDataFrame(df_ocm, geometry=gpd.points_from_xy(df_ocm['AddressInfo.Longitude'], df_ocm['AddressInfo.Latitude']))


# In[45]:


gdf_ocm


# In[46]:


locations = gdf_ocm[['AddressInfo.Latitude', 'AddressInfo.Longitude']]
locationlist = locations.values.tolist()
len(locationlist)
locationlist[7]


# In[47]:


m = folium.Map(location=[52.15484,6.20101],zoom_start=8)
base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
for point in range(0, len(locationlist)):
    folium.Marker(
        location = locationlist[point], 
        popup=gdf_ocm['AddressInfo.Title'][point],
        ).add_to(base_map)
    base_map.add_to(m)

m


# ## 3.3  Statistical model

# #### 3.3.1 grafiek jaren t.o.v. laadpalen in NL

# In[ ]:





# In[ ]:





# In[48]:


df_ocm = df_ocm.assign(aantal_palen=1)


# In[49]:


df_ocm_sorted = df_ocm.sort_values(by=['date_created'])


# In[50]:


df_ocm_sorted['CUMSUM']=df_ocm_sorted['aantal_palen'].cumsum()


# In[51]:


df_ocm_sorted


# In[52]:


plt.scatter(x=df_ocm_sorted['date_created'],y=df_ocm_sorted['CUMSUM'])
plt.xticks(rotation=90)
plt.rcParams["figure.figsize"] = (20,8)
plt.xlabel('Datum geplaatst')
plt.ylabel('Aantal laadpalen in NL')
plt.title('Aantal laadpalen in Nederland over de tijd')
plt.grid()
plt.show()


# Nu wordt er een lineair regressie model opgesteld waar later aannames van kunnen worden aangedaan

# In[53]:


df_ocm_sorted['date_created']=pd.to_datetime(df_ocm_sorted['date_created'])
df_ocm_sorted['date_created']=df_ocm_sorted['date_created'].map(dt.datetime.toordinal)


# In[54]:


X = df_ocm_sorted['date_created'].values.reshape(-1,1)
y = df_ocm_sorted['CUMSUM'].values.reshape(-1,1)


# In[55]:


#de data wordt gescheiden in train en test data om het algoritme te trainen en daarna te testen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[56]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[57]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[58]:


y_pred = regressor.predict(X_test)


# In[59]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.head()


# In[60]:


df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[61]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[62]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Dit is niet een super nauwkeurig model. Wat hier een veroorzaker van zou kunnen zijn is dat de regressie pas begint na 28-9-2019, de eerste datum in de dataset waarin disproportioneel veel datapunten(laadpalen) zijn geplaatst.
# 
# Er wordt vervolgens gekeken naar dezelfde dataset waar de cluster aan waardes zijn weggehaald uit de dataset maar de invloed die zij hebben op de overige waardes blijft behouden

# In[63]:


df_ocm_sorted.head()


# In[ ]:





# In[64]:


df_ocm_sorted2 = df_ocm.sort_values(by=['date_created'])


# In[65]:


df_ocm_sorted2['CUMSUM']=df_ocm_sorted2['aantal_palen'].cumsum()


# In[66]:


df_ocm2 = df_ocm_sorted2.drop(df_ocm_sorted2[df_ocm_sorted2.date_created == '2019-09-28'].index)


# In[67]:


df_ocm2


# In[68]:


plt.scatter(x=df_ocm2['date_created'],y=df_ocm2['CUMSUM'])
plt.xticks(rotation=90)
plt.rcParams["figure.figsize"] = (20,8)
plt.xlabel('Datum geplaatst')
plt.ylabel('Aantal laadpalen in NL')
plt.title('Aantal laadpalen in Nederland over de tijd')
plt.show()


# In[69]:


df_ocm2['date_created']=pd.to_datetime(df_ocm2['date_created'])
df_ocm2['date_created']=df_ocm2['date_created'].map(dt.datetime.toordinal)


# In[70]:


X = df_ocm2['date_created'].values.reshape(-1,1)
y = df_ocm2['CUMSUM'].values.reshape(-1,1)


# In[71]:


#de data wordt gescheiden in train en test data om het algoritme te trainen en daarna te testen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[72]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[73]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[74]:


y_pred = regressor.predict(X_test)


# In[75]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.head()


# In[76]:


df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[77]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[78]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# De Error uitkomsten zijn nu minder dan groot dus het model is wel nauwkeuriger door het aanpassen van de data, alleen hoe nauwkeurig valt nog te bezien. De dataset is nu namelijk aangepast op bevindingen wat de nauwkeurigheid van de uitkomst ook omlaag brengt.

# In[ ]:





# #### 3.3.2

# In[ ]:





# #### 3.3.3

# In[ ]:





# # 3.4 

# #### 3.4.1

# In[ ]:





# #### 3.4.2

# In[ ]:





# # 3.5 

# #### 3.5.1

# In[ ]:





# # 4. Cleaning your dataset
# 

# ## 4.1 Removing Redundant variables

# In[ ]:





# In[ ]:





# Duplicaten zoeken op specifieke kolommen
# 

# In[ ]:





# ## 4.2 Removing Outliers

# In[ ]:





# #### 4.2.1 outlier Detection

# In[ ]:





# In[ ]:





# #### 4.2.2 outliers verwijderen / corrigeren

# In[ ]:





# In[ ]:





# ## 4.3 Corrigeren van technisch onmogelijke waardes

# * spelfouten
# * emailadres zonder @ 
# * datum format verkeerd
# * discontinuiteit sprong in tijds (bijv meter electrische lader / missendetijd) 

# In[ ]:





# In[ ]:





# ## 4.4 data inputation of missing values

# **Missing Values**
# 
# If there are missing values in the Dataset before doing any statistical analysis, we need to handle those missing values.
# There are mainly three types of missing values.
# * MCAR(Missing completely at random): These values do not depend on any other features.
# * MAR(Missing at random): These values may be dependent on some other features.
# * MNAR(Missing not at random): These missing values have some reason for why they are missing.
# 
# Let’s see which columns have missing values in the dataset.
# 
# credits: https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




