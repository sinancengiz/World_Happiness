

```python
# Dependencies
import json
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from pprint import pprint
import os
from scipy.stats import linregress
```


```python
# Store filepath in a variable
file_one = os.path.join("Resources", "2015.csv")
file_two = os.path.join("Resources", "2016.csv")
file_three = os.path.join("Resources", "2017.csv")
file_four = os.path.join("Resources", "fertility_rate.csv")
file_five = os.path.join("Resources", "GDP_per_capita.csv")
```


```python
# Read our Data files with the pandas library
# Not every CSV requires an encoding, but be aware this can come up
happiness_2015_df = pd.read_csv(file_one, encoding="ISO-8859-1")
happiness_2016_df = pd.read_csv(file_two, encoding="ISO-8859-1")
happiness_2017_df = pd.read_csv(file_three, encoding="ISO-8859-1")
fertility_rate_df = pd.read_csv(file_four, encoding="ISO-8859-1")
gdp_per_capita_df = pd.read_csv(file_five, encoding="ISO-8859-1")
democracy_2017_df = pd.read_csv("Resources/Democracy Index by Country 2017.csv")
happiness_2015 = pd.read_csv("Resources/2015.csv")
```


```python
# Show 5 rows and the header 

happiness_2015_df.head()
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
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Standard Error</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.587</td>
      <td>0.03411</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.41978</td>
      <td>0.29678</td>
      <td>2.51738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.561</td>
      <td>0.04884</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.14145</td>
      <td>0.43630</td>
      <td>2.70201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.527</td>
      <td>0.03328</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.48357</td>
      <td>0.34139</td>
      <td>2.49204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.522</td>
      <td>0.03880</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.36503</td>
      <td>0.34699</td>
      <td>2.46531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>North America</td>
      <td>5</td>
      <td>7.427</td>
      <td>0.03553</td>
      <td>1.32629</td>
      <td>1.32261</td>
      <td>0.90563</td>
      <td>0.63297</td>
      <td>0.32957</td>
      <td>0.45811</td>
      <td>2.45176</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rename the collumn
happiness_2015_df = happiness_2015_df.loc[:,["Country","Happiness Score"]]
```


```python
#drop missing values from data frame
gdp_per_capita_df = gdp_per_capita_df.loc[:,["Country Name","2015"]]
gdp_per_capita_df = gdp_per_capita_df.dropna(how='any')
gdp_per_capita_df.head()
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
      <th>Country Name</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>569.577923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>3695.793748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>3934.895394</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>36038.267600</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Arab World</td>
      <td>6435.525509</td>
    </tr>
  </tbody>
</table>
</div>




```python
#change the values of a collumn to numeric
gdp_per_capita_df["2015"] = pd.to_numeric(gdp_per_capita_df["2015"])
gdp_per_capita_df.head()
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
      <th>Country Name</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>569.577923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>3695.793748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>3934.895394</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>36038.267600</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Arab World</td>
      <td>6435.525509</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rename the collumn
gdp_per_capita_df = gdp_per_capita_df.rename(columns={"2015":"GDP Per Capita 2015"})
```


```python
#drop missing values from data frame
fertility_rate_df = fertility_rate_df.loc[:,["Country Name","2015"]]
fertility_rate_df = fertility_rate_df.dropna(how='any')
fertility_rate_df.head()
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
      <th>Country Name</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>1.80100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>4.80200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>5.76600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>1.71400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Arab World</td>
      <td>3.37384</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rename the collumn
fertility_rate_df = fertility_rate_df.rename(columns={"2015":"Birth Rate 2015"})
fertility_rate_df.head()
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
      <th>Country Name</th>
      <th>Birth Rate 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>1.80100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>4.80200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>5.76600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>1.71400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Arab World</td>
      <td>3.37384</td>
    </tr>
  </tbody>
</table>
</div>




```python
#renamethe the collumn
happiness_2015_df = happiness_2015_df.rename(columns={"Country":"Country Name"})
happiness_2015_df.head()
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
      <th>Country Name</th>
      <th>Happiness Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>7.561</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>7.527</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>7.427</td>
    </tr>
  </tbody>
</table>
</div>




```python
#merge hapiness score and birth rate dataframes
merged_dataframe = pd.merge(happiness_2015_df, fertility_rate_df, on="Country Name")
merged_dataframe.head()
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
      <th>Country Name</th>
      <th>Happiness Score</th>
      <th>Birth Rate 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
      <td>1.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>7.561</td>
      <td>1.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>7.527</td>
      <td>1.69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>7.427</td>
      <td>1.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
#merge dataframe with gdp per capita dataframe
merged_dataframe = pd.merge(merged_dataframe, gdp_per_capita_df, on="Country Name")
merged_dataframe.head(5)
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
      <th>Country Name</th>
      <th>Happiness Score</th>
      <th>Birth Rate 2015</th>
      <th>GDP Per Capita 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
      <td>1.54</td>
      <td>82016.02131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>7.561</td>
      <td>1.93</td>
      <td>50734.44360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>7.527</td>
      <td>1.69</td>
      <td>53012.99658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
      <td>1.75</td>
      <td>74498.13764</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>7.427</td>
      <td>1.60</td>
      <td>43335.15911</td>
    </tr>
  </tbody>
</table>
</div>




```python
#set the values to numeric
pd.to_numeric(merged_dataframe["GDP Per Capita 2015"], errors='coerce')
pd.to_numeric(merged_dataframe["GDP Per Capita 2015"], errors='coerce')
#organise the dataframe based on GDp per capita from bigest to smallest
organised_df = merged_dataframe.sort_values("GDP Per Capita 2015", ascending=False)
organised_df.head()
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
      <th>Country Name</th>
      <th>Happiness Score</th>
      <th>Birth Rate 2015</th>
      <th>GDP Per Capita 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Luxembourg</td>
      <td>6.946</td>
      <td>1.500</td>
      <td>101446.78630</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
      <td>1.540</td>
      <td>82016.02131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
      <td>1.750</td>
      <td>74498.13764</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Qatar</td>
      <td>6.611</td>
      <td>1.929</td>
      <td>66346.52267</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ireland</td>
      <td>6.940</td>
      <td>1.940</td>
      <td>62544.63129</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dataframe reorganised based on hapiness score
organised_df = merged_dataframe.sort_values("Happiness Score", ascending=False)
len(organised_df)
```




    137




```python
organised_df_bottom = organised_df.nsmallest(15,"Happiness Score")
organised_df_bottom.head()
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
      <th>Country Name</th>
      <th>Happiness Score</th>
      <th>Birth Rate 2015</th>
      <th>GDP Per Capita 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>Togo</td>
      <td>2.839</td>
      <td>4.517</td>
      <td>551.130835</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Burundi</td>
      <td>2.905</td>
      <td>5.781</td>
      <td>300.676557</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Benin</td>
      <td>3.340</td>
      <td>5.048</td>
      <td>783.947091</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Rwanda</td>
      <td>3.465</td>
      <td>3.967</td>
      <td>710.348391</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Afghanistan</td>
      <td>3.575</td>
      <td>4.802</td>
      <td>569.577923</td>
    </tr>
  </tbody>
</table>
</div>




```python
organised_df_top = organised_df.nlargest(15,"Happiness Score")
organised_df_top.head()
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
      <th>Country Name</th>
      <th>Happiness Score</th>
      <th>Birth Rate 2015</th>
      <th>GDP Per Capita 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
      <td>1.54</td>
      <td>82016.02131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>7.561</td>
      <td>1.93</td>
      <td>50734.44360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>7.527</td>
      <td>1.69</td>
      <td>53012.99658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
      <td>1.75</td>
      <td>74498.13764</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>7.427</td>
      <td>1.60</td>
      <td>43335.15911</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = organised_df.plot(
    kind = 'scatter', 
    x = "GDP Per Capita 2015", 
    y = "Happiness Score", 
    color = 'navy',  
    linewidth = 0.2,
    fontsize = 9
)

ax2 = organised_df_top.plot(
    kind = 'scatter', 
    x = "GDP Per Capita 2015", 
    y = "Happiness Score", 
    color = 'green',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

ax3 = organised_df_bottom.plot(
    kind = 'scatter', 
    x = "GDP Per Capita 2015", 
    y = "Happiness Score", 
    color = 'red',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

# Titles
ax.set_title("Happiness Score vs. GDP Per Capita")
ax.set_xlabel('GDP Per Capita')
ax.set_ylabel('Happiness Score')

# Set min/max of x/y axes
# ax.set_xlim(-0.05,1)
ax.set_ylim(2,8)

# Format colors
ax.set_facecolor('lavender')
ax.grid(color='w', linestyle='-', linewidth=0.7)
for tick in ax.get_xticklines():
    tick.set_color('white')
for tick in ax.get_yticklines():
    tick.set_color('white')
    
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
plt.savefig('Happiness and GDP Per Capita.png')
plt.show()
```


![png](output_17_0.png)



```python
ax = organised_df.plot(
    kind = 'scatter', 
    x = "Happiness Score", 
    y = "Birth Rate 2015", 
    color = 'navy',  
    linewidth = 0.2,
    fontsize = 9
)

ax2 = organised_df_top.plot(
    kind = 'scatter', 
    x = "Happiness Score", 
    y = "Birth Rate 2015", 
    color = 'green',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

ax3 = organised_df_bottom.plot(
    kind = 'scatter', 
    x = "Happiness Score", 
    y = "Birth Rate 2015", 
    color = 'red',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

# Titles
ax.set_title("Happiness Score vs. Birth Rate")
ax.set_xlabel('Happiness Score')
ax.set_ylabel('Birth Rate 2015')

# Set min/max of x/y axes
# ax.set_xlim(-0.05,1)
ax.set_ylim(0,7)

# Format colors
ax.set_facecolor('lavender')
ax.grid(color='w', linestyle='-', linewidth=0.7)
for tick in ax.get_xticklines():
    tick.set_color('white')
for tick in ax.get_yticklines():
    tick.set_color('white')
    
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
plt.savefig('Happiness and Birth Rate.png')
plt.show()
```


![png](output_18_0.png)



```python
happiness_2017_top = happiness_2017_df.nsmallest(15, 'Happiness.Rank')
happiness_2017_bottom = happiness_2017_df.nlargest(15, 'Happiness.Rank')
happiness_2017_top.head()
happiness_2017_bottom.head()
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
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>Central African Republic</td>
      <td>155</td>
      <td>2.693</td>
      <td>2.864884</td>
      <td>2.521116</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018773</td>
      <td>0.270842</td>
      <td>0.280876</td>
      <td>0.056565</td>
      <td>2.066005</td>
    </tr>
    <tr>
      <th>153</th>
      <td>Burundi</td>
      <td>154</td>
      <td>2.905</td>
      <td>3.074690</td>
      <td>2.735310</td>
      <td>0.091623</td>
      <td>0.629794</td>
      <td>0.151611</td>
      <td>0.059901</td>
      <td>0.204435</td>
      <td>0.084148</td>
      <td>1.683024</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Tanzania</td>
      <td>153</td>
      <td>3.349</td>
      <td>3.461430</td>
      <td>3.236570</td>
      <td>0.511136</td>
      <td>1.041990</td>
      <td>0.364509</td>
      <td>0.390018</td>
      <td>0.354256</td>
      <td>0.066035</td>
      <td>0.621130</td>
    </tr>
    <tr>
      <th>151</th>
      <td>Syria</td>
      <td>152</td>
      <td>3.462</td>
      <td>3.663669</td>
      <td>3.260331</td>
      <td>0.777153</td>
      <td>0.396103</td>
      <td>0.500533</td>
      <td>0.081539</td>
      <td>0.493664</td>
      <td>0.151347</td>
      <td>1.061574</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Rwanda</td>
      <td>151</td>
      <td>3.471</td>
      <td>3.543030</td>
      <td>3.398970</td>
      <td>0.368746</td>
      <td>0.945707</td>
      <td>0.326425</td>
      <td>0.581844</td>
      <td>0.252756</td>
      <td>0.455220</td>
      <td>0.540061</td>
    </tr>
  </tbody>
</table>
</div>




```python
# PLOT TREND 1: Happiness score vs. Trust in government, 2017 data
ax = happiness_2017_df.plot(
    kind = 'scatter', 
    x = "Trust..Government.Corruption.", 
    y = "Happiness.Score", 
    color = 'navy',  
    linewidth=0.2,
    fontsize = 9
)

ax2 = happiness_2017_top.plot(
    kind = 'scatter', 
    x = "Trust..Government.Corruption.", 
    y = "Happiness.Score", 
    color = 'green',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

ax3 = happiness_2017_bottom.plot(
    kind = 'scatter', 
    x = "Trust..Government.Corruption.", 
    y = "Happiness.Score", 
    color = 'red',  
    linewidth = 0.2,
    fontsize = 9,
    ax=ax
)

# Titles
ax.set_title("Happiness Score vs. Trust in Government")
ax.set_xlabel('Trust in government')
ax.set_ylabel('Happiness Score')

# Set min/max of x/y axes
# ax.set_xlim(-0.05,1)
ax.set_ylim(0,10)

# Format colors
ax.set_facecolor('lavender')
ax.grid(color='w', linestyle='-', linewidth=0.7)
for tick in ax.get_xticklines():
    tick.set_color('white')
for tick in ax.get_yticklines():
    tick.set_color('white')
    
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')

print("")
print("There is a medium correlation between trust in government and happiness.")

plt.savefig('Happiness and Trust in Government.png')
plt.show()
```

    
    There is a medium correlation between trust in government and happiness.
    


![png](output_20_1.png)



```python
correlation = happiness_2017_df['Trust..Government.Corruption.'].corr(happiness_2017_df['Happiness.Score'])
print("Correlation value: " + str(correlation))
```

    Correlation value: 0.42907973722217224
    


```python
happiness_2017_df_change = happiness_2017_df.iloc[:, [0,2,7,8]]
happiness_2017_df_change.head()
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
      <th>Country</th>
      <th>Happiness.Score</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>7.537</td>
      <td>0.796667</td>
      <td>0.635423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>7.522</td>
      <td>0.792566</td>
      <td>0.626007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>7.504</td>
      <td>0.833552</td>
      <td>0.627163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>7.494</td>
      <td>0.858131</td>
      <td>0.620071</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>7.469</td>
      <td>0.809158</td>
      <td>0.617951</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = happiness_2017_df_change["Health..Life.Expectancy."]
fake = happiness_2017_df_change["Happiness.Score"]
```


```python
(slope, intercept, _, _, _) = linregress(x_axis, fake)
fit = slope * x_axis + intercept

fig, ax = plt.subplots()

fig.suptitle("Happiness Score vs Life Expectancy ", 
fontsize=14, fontweight="bold",)

ax.set_xlim(0, 1)
ax.set_ylim(0, 9)

ax.set_xlabel("Life Expectancy")
ax.set_ylabel("Happiness Score")

ax.plot(x_axis, fake, linewidth=0, marker='o', color='navy', )
ax.plot(x_axis, fit, 'b--')

ax.set_facecolor('lavender')

ax.grid(color='w', linestyle='-', linewidth=0.7)
for tick in ax.get_xticklines():
    tick.set_color('white')
for tick in ax.get_yticklines():
    tick.set_color('white')

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
plt.savefig('Happiness and Life Expectancy.png')
plt.show()
```


![png](output_24_0.png)



```python
#Merge Happiness and Democracy datasets on "country"
combined_data = pd.merge(happiness_2017_df, democracy_2017_df, how='outer', on="Country")

```


```python
#Replace all NaN values with 0
combined_data = combined_data.fillna(0)
combined_data = combined_data[combined_data.Category != 0]
combined_data.head()
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
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
      <th>Rank</th>
      <th>Score</th>
      <th>Electoral Process and Pluralism</th>
      <th>Functioning of Government</th>
      <th>Political Participation</th>
      <th>Political Culture</th>
      <th>Civil Liberties</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1.0</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
      <td>1.0</td>
      <td>9.87</td>
      <td>10.00</td>
      <td>9.64</td>
      <td>10.00</td>
      <td>10.00</td>
      <td>9.71</td>
      <td>Full democracy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2.0</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
      <td>5.0</td>
      <td>9.22</td>
      <td>10.00</td>
      <td>9.29</td>
      <td>8.33</td>
      <td>9.38</td>
      <td>9.12</td>
      <td>Full democracy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3.0</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
      <td>2.0</td>
      <td>9.58</td>
      <td>10.00</td>
      <td>9.29</td>
      <td>8.89</td>
      <td>10.00</td>
      <td>9.71</td>
      <td>Full democracy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4.0</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
      <td>9.0</td>
      <td>9.03</td>
      <td>9.58</td>
      <td>9.29</td>
      <td>7.78</td>
      <td>9.38</td>
      <td>9.12</td>
      <td>Full democracy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5.0</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
      <td>9.0</td>
      <td>9.03</td>
      <td>10.00</td>
      <td>8.93</td>
      <td>7.78</td>
      <td>8.75</td>
      <td>9.71</td>
      <td>Full democracy</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Calculate average "Happiness.Score"
grouped_data = combined_data.groupby(["Category"])
happiness_score = grouped_data.mean()["Happiness.Score"]
happiness_score

happiness_score.values

happiness_score.index
```




    Index(['Authoritarian', 'Flawed democracy', 'Full democracy', 'Hybrid regime'], dtype='object', name='Category')




```python
#Create bar chart
plt.figure(figsize=(12,7))
plt.bar(happiness_score.index, happiness_score.values, align='center', alpha=0.5)
plt.xticks(np.arange(len(happiness_score)),happiness_score.index)
plt.ylim(2, 8)

plt.title('Forms of Government')
plt.ylabel("Happiness Score")
plt.savefig('Happiness and Forms of Government.png')
plt.show()


```


![png](output_28_0.png)



![png](output_28_1.png)



```python
#Calculate average "Happiness.Score"
region_happiness = happiness_2015.groupby(["Region"])
happiness_score = region_happiness.mean()["Happiness Score"]
happiness_score

happiness_score.values

happiness_score.index

#Create bar chart
plt.figure(figsize=(15,7))
plt.bar(happiness_score.index, happiness_score.values, align='center', alpha=0.5)
plt.xticks(np.arange(len(happiness_score)),happiness_score.index)
plt.ylim(2, 8)
plt.xticks(rotation='vertical')


plt.title('World Regions')
plt.ylabel("Happiness Score")
plt.savefig('Happiness and World Regions.png')
plt.show()
```


![png](output_29_0.png)

