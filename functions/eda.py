# Python standar useful libraries
import pandas as pd
from io import StringIO
import numpy as np

# Eda and Statistics
import seaborn as sns
import missingno as msno
import statistics
import statsmodels.api as sm
from scipy import stats

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Custom Functions
from . import stringfunctions # for getting images and string manipulation

# Univariate numerical analysis
def univariate_numeric(eda_variable, data_list, bivariate, cat_var):
    
    # set up the number of figures that are going to be generated
    n_figuras = 3

    #missing values

    # get the missing values
    eda_variable_list = []
    eda_variable_list.append(eda_variable)
    missing_values = pd.DataFrame(data_list,columns=eda_variable_list)

    # set size
    figure(figsize=(8, 8), dpi=80)

    # get the image of the missing values
    plt.figure(n_figuras)
    n_figuras -= 1
    msno.matrix(missing_values, figsize=(8, 8))
    missing_values_image = stringfunctions.get_image()

    # central tendency measures - measures of central location
    mean = statistics.mean(data_list)
    median = statistics.median(data_list)
    mode = statistics.mode(data_list)

    quantiles = ''
    if (len(data_list) > 1): # Check that data is bigger than 1
        quantiles = statistics.quantiles(data_list)

    # measures of spread 
    stdev = ''
    variance = ''
    IQR = ''
    if (len(data_list) > 1): # Check that data is bigger than 1
        stdev = statistics.stdev(data_list)
        variance = statistics.variance(data_list)
        IQR = quantiles[2]-quantiles[0]

    max_value = max(data_list)
    min_value = min(data_list)
    Range = max_value-min_value

    # QQplot
    plt.figure(n_figuras)
    n_figuras -= 1
    sm.qqplot(np.array(data_list), line='s')
    QQplot_image = stringfunctions.get_image()

    # Anderson-Darling test - Normality test
    sw_test = ''
    if (len(data_list) > 2): # Check that data is bigger than 2
        sw_test = stats.anderson(data_list, dist='norm')

    # Skewness and Kurtosis
    skewness_test = stats.skew(data_list)
    kurtosis_test = stats.kurtosis(data_list)

    # return all the useful local variables (tuple format)
    
    if (bivariate == False):
        
        # histogram
        plt.figure(n_figuras)
        n_figuras -= 1
        sns.displot(data_list)
        histogram = stringfunctions.get_image()
        
        #boxplot
        plt.clf()
        plt.figure(n_figuras)
        n_figuras -= 1
        sns.boxplot(x=data_list)
        box_plot_image = stringfunctions.get_image()
        plt.clf()
        
        
    else:
 
        #boxplot
        plt.clf()
        plt.figure(n_figuras)
        plt.title("Boxplot con el valor de la variable catégorica " + str(cat_var))
        n_figuras -= 1
        sns.boxplot(x=data_list)
        box_plot_image = stringfunctions.get_image()
        plt.clf()
        
        # histogram
        plt.figure(figsize=(10,6))
        plt.figure(n_figuras)
        n_figuras -= 1
        g = sns.displot(data_list)
        g.fig.subplots_adjust(top=.95, right=.9)
        
        plt.title("Histograma - valor de la variable catégorica " + str(cat_var))
        histogram = stringfunctions.get_image()
        
        
    return missing_values_image, histogram, mean, median, mode, quantiles, stdev, variance, max_value, min_value, Range, IQR, box_plot_image, QQplot_image, sw_test, skewness_test, kurtosis_test

# Univariate categorical analysis
def univariate_categorical(eda_variable, data_list):
    
    n_figuras = 1
    
    # set size
    figure(figsize=(8, 8), dpi=80)

    #missing values
    
    # get the missing values
    eda_variable_list = []
    eda_variable_list.append(eda_variable)
    missing_values = pd.DataFrame(data_list,columns=eda_variable_list)
    
    # get the image of the missing values
    plt.figure(n_figuras)
    n_figuras -= 1
    msno.matrix(missing_values, figsize=(8, 8))
    missing_values_image = stringfunctions.get_image()

    #bar plot - could be replaced or add a sector plot
    keys, counts = np.unique(data_list, return_counts=True)
    plt.figure(n_figuras)
    n_figuras -= 1
    plt.bar(keys, counts)
    bar_plot_image = stringfunctions.get_image()

    # frequency tables
    absolute = pd.Series(data_list).value_counts()
    relative = (pd.Series(data_list).value_counts())/len(data_list)
    
    return missing_values_image, bar_plot_image, absolute.to_dict(), relative.to_dict()

def bivariate_numeric(data_list, data_list2):
    
    # set size
    figure(figsize=(8, 8), dpi=80)
    
    # Scatter plot with regression line
    plt.figure(0)
    plt.plot(data_list, data_list2, 'o')
    m, b = np.polyfit(data_list, data_list2, 1)
    plt.plot(np.array(data_list), m*np.array(data_list) + b)
    scatter_plot = stringfunctions.get_image()

    # Correlation coefficient
    Pearson_coeff = np.corrcoef(data_list, data_list2)
    
    return scatter_plot, Pearson_coeff

def bivariate_categorical(data_list, data_list2):
    
    # set size
    figure(figsize=(8, 8), dpi=80)
    
    # Contingency table
    data_crosstab = pd.crosstab(index=[data_list], columns=[data_list2], margins = False)
    data_crosstab_df = (pd.DataFrame(data_crosstab)).to_html

    # Pearson Chi-Squared
    chi2 = stats.chi2_contingency(np.array(data_crosstab))

    # Build a dataframe with the values of both lists
    df_2cat = pd.DataFrame()
    df_2cat['var1'] = data_list
    df_2cat['var2'] = data_list2
    
    # Bar plot of both variables in two dimensions with colours
    plt.figure(0)
    sns.histplot(binwidth=0.5, x="var1", hue="var2", data=df_2cat, stat="count", multiple="stack")
    bar_plot_2cat_image = stringfunctions.get_image()
    
    return data_crosstab_df, chi2, bar_plot_2cat_image

def bivariate_numeric_categorical(eda_variable, eda_variable2, numeric, numeric2, data, data2, model):
    
    # set size
    figure(figsize=(8, 8), dpi=80)
    
    # Filter by categorical value and get the most important measures in single variable numeric analysis
  
    # Get which is the numeric/categorical variable and the unique values of data
    if (not(numeric) and numeric2):
        var_numeric = eda_variable2
        var_categorical = eda_variable
        unique_var = np.unique(data)

    else:
        var_numeric = eda_variable
        var_categorical = eda_variable2
        unique_var = np.unique(data2)

    
    # Declare empty lists for the filtered data 
    histogram_list = []
    mean_list = []
    median_list = []
    mode_list = []
    quantiles_list = []
    stdev_list = []
    variance_list = []
    max_value_list = []
    min_value_list = []
    Range_list = []
    IQR_list = []
    box_plot_list = []
    sw_list = []

    # set number of figures
    n_figuras = len(unique_var)*2

    # Iterate over all unique values
    for cat_var in unique_var:
        
        # Clear plot window
        plt.clf()

        # declare the numeric list 
        data_list_numeric = []

        # automatic filtering
        filter_kwargs = {
            "{}__contains".format(var_categorical): cat_var
        }
        data_numeric = model.objects.filter(**filter_kwargs).values_list(var_numeric)

        # build the numeric list 
        for element in data_numeric:
            data_list_numeric.append(float(element[0]))  
            
        missing_values_image, histogram, mean, median, mode, quantiles, stdev, variance, max_value, min_value, Range, IQR, box_plot_image, QQplot_image, sw_test, skewness_test, kurtosis_test = univariate_numeric(var_numeric, data_list_numeric, True, cat_var)

        # Fill the lists
        
        histogram_list.append(histogram)
        mean_list.append((cat_var,mean))
        median_list.append((cat_var,median))
        mode_list.append((cat_var,mode))
        max_value_list.append((cat_var,max_value))
        min_value_list.append((cat_var,min_value))
        Range_list.append((cat_var,Range))
        box_plot_list.append(box_plot_image)
        
        # Check if the results are valid and fill the lists
        if (len(data_numeric) > 1):
            quantiles_list.append((cat_var,quantiles))
            stdev_list.append((cat_var,stdev))
            variance_list.append((cat_var,variance))
            IQR_list.append((cat_var,IQR))

        if (len(data_numeric) > 2):
            sw_list.append((cat_var,sw_test))

    return histogram_list, mean_list, median_list, mode_list, quantiles_list, stdev_list, variance_list, max_value_list, min_value_list, Range_list, IQR_list, box_plot_list, sw_list