# previous operations with classification (CV, etc.)
from . import sparkclassificationalgorithms
from . import stringfunctions

# For metrics calculation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import math

# Models
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor

# Evaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Grid builder
from pyspark.ml.tuning import ParamGridBuilder

# plot
import matplotlib.pyplot as plt

# Function for geting the estimator and the grid

def create_estimator_grid(algorithm_param, max_depth_param, max_bins_param, min_instances_per_node_param, min_info_gain_param, max_memory_in_mb_param, impurity_param, subsamplingRate_param, numTrees_param, featureSubsetStrategy_param, bootstrap_param, maxIter_param, lossType_param, validationTol_param, stepSize_param, minWeightFractionPerNode_param, logger_param):
    
    if (algorithm_param == "DecisionTreeRegression"):
        
        # Create the Decision Tree estimator and indicate the relevant columns
        est = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param, minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, impurity=impurity_param)

        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).build()
        
    elif (algorithm_param == "RandomForestRegression"):
        
        # Create the Decision Tree estimator and indicate the relevant columns
        est = RandomForestRegressor(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param, minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, impurity=impurity_param, subsamplingRate=subsamplingRate_param, numTrees=numTrees_param, featureSubsetStrategy=featureSubsetStrategy_param, bootstrap=bootstrap_param)

        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).addGrid(est.subsamplingRate, [subsamplingRate_param]).addGrid(est.featureSubsetStrategy, [featureSubsetStrategy_param]).addGrid(est.numTrees, [numTrees_param]).addGrid(est.bootstrap, [bootstrap_param]).build()
        
    elif (algorithm_param == "GBTRegression"):
        
        # Create the Gradient Boosted Tree estimator and indicate the relevant columns //// Commented impurity. Bug found
        est = GBTRegressor(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param,
                                    minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, subsamplingRate = subsamplingRate_param,
                                    featureSubsetStrategy = featureSubsetStrategy_param, maxIter = maxIter_param, lossType = lossType_param, 
                                    validationTol = validationTol_param,
                                    stepSize = stepSize_param, minWeightFractionPerNode = minWeightFractionPerNode_param)
        
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).addGrid(est.subsamplingRate, [subsamplingRate_param]).addGrid(est.featureSubsetStrategy, [featureSubsetStrategy_param]).addGrid(est.maxIter, [maxIter_param]).addGrid(est.lossType, [lossType_param]).addGrid(est.validationTol, [validationTol_param]).addGrid(est.stepSize, [stepSize_param]).addGrid(est.minWeightFractionPerNode, [minWeightFractionPerNode_param]).build()
        
    return est, grid

######################################### REGRESSION ALGORITHMS #########################################

def regressors(algorithm_param = '', df = '', parallelism_param = '', n_folds_param = '', max_depth_param = '', max_bins_param = '', min_instances_per_node_param = '', min_info_gain_param = '', max_memory_in_mb_param = '', impurity_param = '', subsamplingRate_param = '', numTrees_param = '', featureSubsetStrategy_param = '', bootstrap_param = '', maxIter_param = '', lossType_param = '', validationTol_param = '', stepSize_param = '', minWeightFractionPerNode_param = '', logger_param = ''):
    
    # create the estimator and grid, depending on the algorithm
    est, grid = create_estimator_grid(algorithm_param, max_depth_param, max_bins_param, min_instances_per_node_param, min_info_gain_param, max_memory_in_mb_param, impurity_param, subsamplingRate_param, numTrees_param, featureSubsetStrategy_param, bootstrap_param, maxIter_param, lossType_param, validationTol_param, stepSize_param, minWeightFractionPerNode_param, logger_param)
    
    # Create evaluator (Regression evaluator, using metric RMSE)
    evaluator = RegressionEvaluator()
    
    # Create the cross validator using the estimator, the grid, the evaluator and the selected number of cores and folds 
    cv  = sparkclassificationalgorithms.CrossValidatorBestModelTraining(estimator=est, estimatorParamMaps=grid, evaluator=evaluator, parallelism=parallelism_param,
                                                                        numFolds = n_folds_param)
    
    # Fit the CV
    cvModel, metrics, bestTrainingModel_train, bestTrainingModel_test, bestIndex2 = cv.fit(df)
    
    #get the best model found in CV
    bestModel = cvModel.bestModel
    
    # Get the predictions and labels of the best training model found in CV
    predictions = bestModel.transform(bestTrainingModel_test)
    
    # get predictions, labels and probabilities as lists
    
    # change it to pandas df
    predictions_pandas = predictions.select("label", "prediction").toPandas()
    
    # get predictions and labels
    predictions_list = list(predictions_pandas["prediction"])
    labels_list = list(predictions_pandas["label"])
    
    logger_param.info("predictions")
    logger_param.info(predictions_list)
    logger_param.info("labels")
    logger_param.info(labels_list)
    
    ######### Get the most relevant results #########
        
    # Number of folds
    num_folds = n_folds_param
    
    # average RMSE of each validation set of each fold, not training set
    RMSEMean = cvModel.avgMetrics

    # MAE - Mean Absolute Error
    MAE = mean_absolute_error(labels_list, predictions_list)

    # MAPE - Mean absolute percentage error
    MAPE = mean_absolute_percentage_error(labels_list, predictions_list)

    # MSE - Mean Squared Error
    MSE = mean_squared_error(labels_list, predictions_list)

    # RMSE - Root Mean Squared Error, sqrt(MSE)
    RMSE = math.sqrt(MSE)

    # R2 coefficient, coefficient of determination
    R2 = r2_score(labels_list, predictions_list)
    
    # Plot with predicted and actual values
    labels_list_ordered, predictions_list_ordered = zip(*sorted(zip(labels_list, predictions_list)))
    
    plt.clf()
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(list(range(len(labels_list_ordered))),labels_list_ordered, marker='o', color='b', label="Valores reales")
    plt.plot(list(range(len(predictions_list_ordered))),predictions_list_ordered, marker='o', color='orange', label="Valores predichos")
    plt.ylabel('Valor de regresión')
    plt.legend()
    plt.title("Gráfica de regresión, valores reales y predichos de la variable (ordenados por valor real)")
    
    Regression_plot = stringfunctions.get_image()
    
    return num_folds, RMSEMean, MAE, MAPE, MSE, RMSE, R2, Regression_plot