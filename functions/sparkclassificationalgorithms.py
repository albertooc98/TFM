from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, _parallelFitTasks, CrossValidatorModel
import pandas as pd
import matplotlib.pyplot as plt
from . import stringfunctions
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import pool
from pyspark.sql.functions import rand
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn

# clasificación basada en https://spark.apache.org/docs/2.2.0/mllib-evaluation-metrics.html#binary-classification y en
# https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa (binaria)

# testing Cross Validator Verbose
class CrossValidatorBestModelTraining(CrossValidator):
    
    # Just included, not modified
    def _kFold(self, dataset):
        nFolds = self.getOrDefault(self.numFolds)
        foldCol = False # Changed to False

        datasets = []
        if not foldCol:
            # Do random k-fold split.
            seed = self.getOrDefault(self.seed)
            h = 1.0 / nFolds
            randCol = self.uid + "_rand"
            df = dataset.select("*", rand(seed).alias(randCol))
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
                validation = df.filter(condition)
                train = df.filter(~condition)
                datasets.append((train, validation))
        else:
            # Use user-specified fold numbers.
            def checker(foldNum):
                if foldNum < 0 or foldNum >= nFolds:
                    raise ValueError(
                        "Fold number must be in range [0, %s), but got %s." % (nFolds, foldNum))
                return True

            checker_udf = UserDefinedFunction(checker, BooleanType())
            for i in range(nFolds):
                training = dataset.filter(checker_udf(dataset[foldCol]) & (col(foldCol) != lit(i)))
                validation = dataset.filter(
                    checker_udf(dataset[foldCol]) & (col(foldCol) == lit(i)))
                if training.rdd.getNumPartitions() == 0 or len(training.take(1)) == 0:
                    raise ValueError("The training data at fold %s is empty." % i)
                if validation.rdd.getNumPartitions() == 0 or len(validation.take(1)) == 0:
                    raise ValueError("The validation data at fold %s is empty." % i)
                datasets.append((training, validation))

        return datasets

    # fit modified
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics = [0.0] * numModels
        
        # metrics of each fold
        metrics2 = [0.0] * nFolds

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)
        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)
                metrics2[i] += metric
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
            
            # get the best fold
            bestIndex2 = np.argmax(metrics2)
        else:
            bestIndex = np.argmin(metrics)
            
            # get the best fold
            bestIndex2 = np.argmin(metrics2)
        
        # get the best training model, test and training datasets
        bestTrainingModel_train = datasets[bestIndex2][0]
        bestTrainingModel_test = datasets[bestIndex2][1]
        
        # bestModel is going to be Best training Model
        bestModel = est.fit(bestTrainingModel_train, epm[bestIndex])
        
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels)), metrics2, bestTrainingModel_train, bestTrainingModel_test, bestIndex2
    
# function for plotting the roc curve
def get_roc_curve(probability_0_param, probability_1_param, labels_param, tv_map_param):
    
    # get title of the plots
    title_0 = "ROC curve with positive class 0 - '" + str(tv_map_param[0]) + "'"
    title_1 = "ROC curve with positive class 1 - '" + str(tv_map_param[1]) + "'"
    
    # get the roc curve where the label 0 is the positive case POSITIVE-NEGATIVE
    image_0 = get_roc_curve_image(probability_0_param, labels_param.count(0), labels_param.count(1), labels_param, 0, title_0)
    # get the roc curve where the label 1 is the positive case POSITIVE-NEGATIVE
    image_1 = get_roc_curve_image(probability_1_param, labels_param.count(1), labels_param.count(0), labels_param, 1, title_1)
    
    return image_0, image_1

def get_roc_curve_image(prob_param, P, N, labels_param, positive_class_param, title_param):
    
    # sorted list
    prob = prob_param.copy()
    prob.sort(reverse=True)
    
    # initialize the rest of parameters
    FP = 0
    TP = 0
    R = []
    f_prev = -1
    
    # for every instance O(n)
    for i in range(len(prob)):
        
        # update previous score value
        if (f_prev != prob[i]):
            R.append([ FP/N, TP/P ])
            f_prev = prob[i]
        
        # update TP and FP values
        if (labels_param[i] == positive_class_param):
            TP += 1
        else:
            FP += 1
    
    # append last value of the ROC curve
    R.append([1,1])
    
    # get the FPR and TPR
    FPR = []
    TPR = []
    for elem in R:
        FPR.append(elem[0])
        TPR.append(elem[1])
    
    # Build the plot
#     plt.figure(figsize=(8, 8), dpi=80)
    plt.clf()
    plt.title(title_param)
    plt.plot(FPR,TPR)
    plt.plot([0,1],[0,1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
    # annotations
    for x,y,score in zip(FPR,TPR,(["Infinity"]+prob+["-Infinity"])):

        if (str(type(score)) != "<class 'str'>"):
            label = "{:.5f}".format(score)
        else:
            label = "{lab}".format(lab = score)

        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
    
    # get the image
    ROC_image = stringfunctions.get_image()
    
    return ROC_image

def get_rates(predictions, labels, positive_class_label, negative_class_label):
    
    # all rates
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(predictions)): 
        if labels[i]==predictions[i]==positive_class_label:
            TP += 1
        if predictions[i]==positive_class_label and labels[i]!=predictions[i]:
            FP += 1
        if labels[i]==predictions[i]==negative_class_label:
            TN += 1
        if predictions[i]==negative_class_label and labels[i]!=predictions[i]:
            FN += 1

    # precision
    if (TP > 0):
        precision = TP/(TP+FP)
    else:
        precision = 0
    
    # recall or sensitivity or TP Rate 0.5 threshold (from confusion matrix)
    recall = TP/labels.count(positive_class_label)
    
    # FP Rate 0.5 threshold (from confusion matrix)
    FPR = FP/labels.count(negative_class_label)
    
    # TN Rate or specificity
    TNR = 1-FPR
    
    # F1 measure or F1 Score
    F1 = TP/(TP+1/2*(FP+FN))
    
    return TP, FP, TN, FN, precision, recall, FPR, TNR, F1

def get_measures(predictions, labels):
    
    # accuracy
    accuracy = sum(1 for x,y in zip(predictions,labels) if x == y) / len(predictions)
    
    # confusion matrix
    plt.clf()
    sn.heatmap(confusion_matrix(labels, predictions), annot=True).set(xlabel="Predictions", ylabel = "Labels")
    cm = stringfunctions.get_image()
    
    # normalized confusion matrix
    plt.clf()
    sn.heatmap(confusion_matrix(labels, predictions, normalize='pred'), annot=True).set(xlabel="Predictions", ylabel = "Labels")
    cm_norm = stringfunctions.get_image()

    # all rates, precision, recall, FPR, TNR, F1 score
    
    # with 0 as positive class
    TP_0, FP_0, TN_0, FN_0, precision_0, recall_0, FPR_0, TNR_0, F1_0 = get_rates(predictions, labels, 0, 1)
    
    # with 1 as positive class
    TP_1, FP_1, TN_1, FN_1, precision_1, recall_1, FPR_1, TNR_1, F1_1 = get_rates(predictions, labels, 1, 0)
    
    # we keep the measures with best F1 Score
    
    return accuracy, cm, cm_norm, precision_0, recall_0, FPR_0, TNR_0, F1_0, precision_1, recall_1, FPR_1, TNR_1, F1_1
    

############################ Multinomial Logistic Regression ############################

def binary_log_regression_classifier_cv(df, maxIter_param, parallelism_param, n_folds_param):
    
    # Create the Logistic Regresion model and indicate the relevant columns
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    
    # Create the grid with the parameters that are going to be tested
    grid = ParamGridBuilder().addGrid(lr.maxIter, [maxIter_param]).build()
    
    # Create evaluator (Binary classification default evaluator, using numBins = 1000 and metric AUC)
    evaluator = BinaryClassificationEvaluator()
    
    # Create the cross validator using the estimator, the grid, the evaluator and the selected number of cores and folds 
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, parallelism=parallelism_param, numFolds = n_folds_param)
    
    # Fit the CV
    cvModel = cv.fit(df)
    
    ######### Get the most relevant results #########
     # thigs that we have to keep in a binary LRM: beta coefficients, AUC, ROC curve (FPR vs TPR)-ver de donde los coge (supongo que del threshold luce que sí porque en scala lo usa así), precision-recall curve y accuracy
        
    # Number of folds
    num_folds = n_folds_param

    # average AUC of each validation set of each fold, not training set
    AUC_mean = cvModel.avgMetrics
    
    # get the summary of the best found model
    trainingSummary = cvModel.bestModel.summary
    
    # coefficients of the best linear regression model
    coeff = cvModel.bestModel.coefficients
    
    # Best training model AUC
    AUC_best = trainingSummary.areaUnderROC
    
    # Best training model Accuracy
    accuracy_best = trainingSummary.accuracy
    
    # Best training model ROC curve
    ROC_best = trainingSummary.roc
    
    # Best training model PR (precision-recall)
    PR_best = trainingSummary.pr
    
    # get the images
    
    # coefficients
    coeff_sort = np.sort(coeff)
    plt.figure(2)
    plt.title("Model's Beta coefficients (sorted)")
    plt.ylabel("Beta Coefficients")
    plt.plot(coeff_sort)
    coeff_image = stringfunctions.get_image()
    
    # ROC curve
    ROC_pd = ROC_best.toPandas()
    plt.figure(1)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(ROC_pd['FPR'], ROC_pd['TPR'])
    ROC_image = stringfunctions.get_image()
    
    # Precision-Recall curve
    PR_pd = PR_best.toPandas()
    plt.figure(0)
    plt.title("PR Curve")
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(PR_pd['recall'], PR_pd['precision'])
    PR_image = stringfunctions.get_image()
    
    return num_folds, AUC_mean, coeff, AUC_best, accuracy_best, ROC_best, PR_best, coeff_image, ROC_image, PR_image

############################ Multinomial Logistic Regression ############################

def multiclass_log_regression_classifier_cv(df, maxIter_param, parallelism_param, n_folds_param):
    # Create the Logistic Regresion model and indicate the relevant columns
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    
    # Create the grid with the parameters that are going to be tested
    grid = ParamGridBuilder().addGrid(lr.maxIter, [maxIter_param]).build()
    
    # Create evaluator (Multiclass classification default evaluator, using default metric F1-Score and rest of default parameters: metricLabel=0.0, beta=1.0 and eps=1e-15)
    evaluator = MulticlassClassificationEvaluator()
    
    # Create the cross validator using the estimator, the grid, the evaluator and the selected number of cores and folds 
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, parallelism=parallelism_param, numFolds = n_folds_param)
    
    # Fit the CV
    cvModel = cv.fit(df)
    
    ######### Get the most relevant results #########
    
     # thigs that we have to keep in a multiclass LRM: averahe metrics, matrix of coefficients, metrics per-label, accuracy and weighted  falsePositiveRate, truePositiveRate, fMeasure, precision, recall
        
    # Number of folds
    num_folds = n_folds_param

    # average F1 of each validation set of each fold, not training set
    F1_mean = cvModel.avgMetrics
    
    # coefficients matrix of the best linear regression model
    coeff_matrix = cvModel.bestModel.coefficientMatrix
    
    # get the summary of the best founded model
    trainingSummary = cvModel.bestModel.summary
    
    # best training model accuracy
    accuracy_best = trainingSummary.accuracy
    
    ## Best training model metrics by label ##

    # F1 Score by label
    F1_by_label = []
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        F1_by_label.append([i,f])
    
    # False Positive Rate by label
    FPR_by_label = []
    for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
        FPR_by_label.append([i,rate])
    
    # True Positive Rate by label
    TPR_by_label = []
    for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
        TPR_by_label.append([i,rate])
    
    # Precision by label
    precision_by_label = []
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        precision_by_label.append([i,rate])
    
    # Recall by label
    recall_by_label = []
    for i, rec in enumerate(trainingSummary.recallByLabel):
        recall_by_label.append([i,rate])
    
    ## Best training model Weighted measures ##
    
    # Weigthed F1 Score
    F1_weighted = trainingSummary.weightedFMeasure()
    
    # Weigthed False Positive Rate
    FPR_weighted = trainingSummary.weightedFalsePositiveRate
    
    # Weigthed True Positive Rate
    TPR_weighted = trainingSummary.weightedTruePositiveRate
    
    # Weigthed Precision 
    precision_weighted = trainingSummary.weightedPrecision
    
    # Weigthed Recall
    recall_weighted = trainingSummary.weightedRecall

    return num_folds, F1_mean, coeff_matrix, accuracy_best, F1_by_label, FPR_by_label, TPR_by_label, precision_by_label, recall_by_label, F1_weighted, FPR_weighted, TPR_weighted, precision_weighted, recall_weighted

# function for building the estimator and grid
def create_estimator_grid(algorithm_param, parallelism_param, n_folds_param, max_depth_param, max_bins_param, min_instances_per_node_param, min_info_gain_param, max_memory_in_mb_param, impurity_param, subsamplingRate_param, featureSubsetStrategy_param, numTrees_param, bootstrap_param, maxIter_param, lossType_param, validationTol_param, stepSize_param, minWeightFractionPerNode_param, smoothing_param, modelType_param, aggregationDepth_param, regParam_param, standardization_param, tol_param):
    
    if (algorithm_param == "BinaryDecisionTree" or algorithm_param == "MulticlassDecisionTree"):
        
        # Create the Decision Tree estimator and indicate the relevant columns
        est = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param, minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, impurity=impurity_param)

        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).build()
        
    elif (algorithm_param == "BinaryRandomForest" or algorithm_param == "MulticlassRandomForest"):
        
        # Create the Random Forest estimator and indicate the relevant columns
        est = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param,
                                    minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, impurity=impurity_param, subsamplingRate = subsamplingRate_param,
                                    featureSubsetStrategy = featureSubsetStrategy_param, numTrees = numTrees_param, bootstrap = bootstrap_param)

        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).addGrid(est.subsamplingRate, [subsamplingRate_param]).addGrid(est.featureSubsetStrategy, [featureSubsetStrategy_param]).addGrid(est.numTrees, [numTrees_param]).addGrid(est.bootstrap, [bootstrap_param]).build()
        
    elif (algorithm_param == "BinaryGBT"):
        
        # Create the Gradient Boosted Tree estimator and indicate the relevant columns
        est = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=max_depth_param, maxBins=max_bins_param, minInstancesPerNode=min_instances_per_node_param,
                                    minInfoGain=min_info_gain_param, maxMemoryInMB=max_memory_in_mb_param, impurity=impurity_param, subsamplingRate = subsamplingRate_param,
                                    featureSubsetStrategy = featureSubsetStrategy_param, maxIter = maxIter_param, lossType = lossType_param, validationTol = validationTol_param,
                                    stepSize = stepSize_param, minWeightFractionPerNode = minWeightFractionPerNode_param)
        
        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxDepth, [max_depth_param]).addGrid(est.maxBins, [max_bins_param]).addGrid(est.minInstancesPerNode, [min_instances_per_node_param]).addGrid(est.minInfoGain, [min_info_gain_param]).addGrid(est.maxMemoryInMB, [max_memory_in_mb_param]).addGrid(est.impurity, [impurity_param]).addGrid(est.subsamplingRate, [subsamplingRate_param]).addGrid(est.featureSubsetStrategy, [featureSubsetStrategy_param]).addGrid(est.maxIter, [maxIter_param]).addGrid(est.lossType, [lossType_param]).addGrid(est.validationTol, [validationTol_param]).addGrid(est.stepSize, [stepSize_param]).addGrid(est.minWeightFractionPerNode, [minWeightFractionPerNode_param]).build()
        
    elif (algorithm_param == "BinaryNaiveBayes" or algorithm_param == "MulticlassNaiveBayes"):
        
        # Create the Gradient Boosted Tree estimator and indicate the relevant columns
        est = NaiveBayes(featuresCol = 'features', labelCol = 'label', smoothing = smoothing_param, modelType = modelType_param)
        
        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.smoothing, [smoothing_param]).addGrid(est.modelType, [modelType_param]).build()
        
    elif (algorithm_param == "BinaryLinearSVC"):
        
        # Create the Gradient Boosted Tree estimator and indicate the relevant columns
        est = LinearSVC(featuresCol = 'features', labelCol = 'label', maxIter = maxIter_param, regParam = regParam_param, standardization = standardization_param, aggregationDepth = aggregationDepth_param, tol = tol_param)
        
        # Create the grid with the parameters that are going to be tested
        grid = ParamGridBuilder().addGrid(est.maxIter, [maxIter_param]).addGrid(est.regParam, [regParam_param]).addGrid(est.standardization, [standardization_param]).addGrid(est.aggregationDepth, [aggregationDepth_param]).addGrid(est.tol, [tol_param]).build()
         
    return est, grid
    

################################# REST OF BINARY CLASSIFIERS #################################

def binary_classifiers(algorithm_param = '', df = '', parallelism_param = '', n_folds_param = '', max_depth_param = '', max_bins_param = '', min_instances_per_node_param = '', min_info_gain_param = '', max_memory_in_mb_param = '', impurity_param = '', tv_map_param = '', subsamplingRate_param = '', featureSubsetStrategy_param = '', numTrees_param = '', bootstrap_param  = '', maxIter_param = '', lossType_param = '', validationTol_param = '', stepSize_param = '', minWeightFractionPerNode_param = '', smoothing_param = '', modelType_param = '', aggregationDepth_param = '', regParam_param = '', standardization_param = '', tol_param = ''):
    
    # create the estimator and grid, depending on the algorithm
    est, grid = create_estimator_grid(algorithm_param, parallelism_param, n_folds_param, max_depth_param, max_bins_param, min_instances_per_node_param, min_info_gain_param, max_memory_in_mb_param, impurity_param, subsamplingRate_param, featureSubsetStrategy_param, numTrees_param, bootstrap_param, maxIter_param, lossType_param, validationTol_param, stepSize_param, minWeightFractionPerNode_param, smoothing_param, modelType_param, aggregationDepth_param, regParam_param, standardization_param, tol_param)
    
    # Create evaluator (Binary classification default evaluator, using numBins = 1000 and metric AUC)
    evaluator = BinaryClassificationEvaluator()
    
    # Create the cross validator using the estimator, the grid, the evaluator and the selected number of cores and folds 
    cv  = CrossValidatorBestModelTraining(estimator=est, estimatorParamMaps=grid, evaluator=evaluator, parallelism=parallelism_param, numFolds = n_folds_param)
    
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
    
    ######### Get the most relevant results #########
        
    # Number of folds
    num_folds = n_folds_param

    # average AUC of each validation set of each fold, not training set
    AUC_mean = cvModel.avgMetrics
    
    # Best training model AUC
    AUC_best = metrics[bestIndex2]
    
    # get the rest of measures:
    accuracy, cm, cm_norm, precision_0, recall_0, FPR_0, TNR_0, F1_0, precision_1, recall_1, FPR_1, TNR_1, F1_1 = get_measures(predictions_list, labels_list)
    
    # get the roc curve in case it is a probabilistic classifier or a discrete one with probabilities
    if (algorithm_param != "BinaryGBT" and algorithm_param != "BinaryLinearSVC"):
    
        predictions_pandas["probability"] = predictions.select("probability").toPandas()['probability']
    
        # initialize and get the lists from pandas elements
        prob_0_list = []
        prob_1_list = []

        for elem in predictions_pandas["probability"]:
            prob_0_list.append(elem[0])
            prob_1_list.append(elem[1])
        
        # get manually the roc curve using the algorithm of the article https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf
        ROC_image_0, ROC_image_1 = get_roc_curve(prob_0_list, prob_1_list, labels_list, tv_map_param)
    
        return num_folds, AUC_mean, AUC_best, accuracy, cm, cm_norm, precision_0, recall_0, FPR_0, TNR_0, F1_0, precision_1, recall_1, FPR_1, TNR_1, F1_1, ROC_image_0, ROC_image_1
        
    return num_folds, AUC_mean, AUC_best, accuracy, cm, cm_norm, precision_0, recall_0, FPR_0, TNR_0, F1_0, precision_1, recall_1, FPR_1, TNR_1, F1_1

def multiclass_classifiers(algorithm_param = '', df = '', parallelism_param = '', n_folds_param = '', max_depth_param = '', max_bins_param = '', min_instances_per_node_param = '', min_info_gain_param = '', max_memory_in_mb_param = '', impurity_param = '', subsamplingRate_param = '', featureSubsetStrategy_param = '', numTrees_param = '', bootstrap_param = '', maxIter_param = '', lossType_param = '', validationTol_param = '', stepSize_param = '', minWeightFractionPerNode_param = '', smoothing_param = '', modelType_param = '', aggregationDepth_param = '', regParam_param = '', standardization_param = '', tol_param = ''):

     # create the estimator and grid, depending on the algorithm
    est, grid = create_estimator_grid(algorithm_param, parallelism_param, n_folds_param, max_depth_param, max_bins_param, min_instances_per_node_param, min_info_gain_param, max_memory_in_mb_param, impurity_param, subsamplingRate_param, featureSubsetStrategy_param, numTrees_param, bootstrap_param, maxIter_param, lossType_param, validationTol_param, stepSize_param, minWeightFractionPerNode_param, smoothing_param, modelType_param, aggregationDepth_param, regParam_param, standardization_param, tol_param)
    
    # Create evaluator (Multiclass classification default evaluator, using default metric F1-Score and rest of default parameters: metricLabel=0.0, beta=1.0 and eps=1e-15)
    evaluator = MulticlassClassificationEvaluator()
    
    # Create the cross validator using the estimator, the grid, the evaluator and the selected number of cores and folds 
    cv  = CrossValidatorBestModelTraining(estimator=est, estimatorParamMaps=grid, evaluator=evaluator, parallelism=parallelism_param, numFolds = n_folds_param)
    
    # Fit the CV
    cvModel, metrics_cv, bestTrainingModel_train, bestTrainingModel_test, bestIndex2 = cv.fit(df)
    
    #get the best model found in CV
    bestModel = cvModel.bestModel
    
    # Get the predictions and labels of the best training model found in CV
    predictions = bestModel.transform(bestTrainingModel_test)
    
    # get predictions, labels and probabilities as lists
    
    # change it to pandas df
    predictions_pandas = predictions.select("probability", "label", "prediction").toPandas()
    
    # get measures and results
    
    # Number of folds
    num_folds = n_folds_param

    # average F1 of each validation set of each fold, not training set
    F1_mean = cvModel.avgMetrics
    
    # confusion matrix
    plt.clf()
    sn.heatmap(confusion_matrix(predictions_pandas["label"], predictions_pandas["prediction"]), annot=True).set(xlabel="Predicted", ylabel = "Labels")
    cm = stringfunctions.get_image()
    
    # normalized confusion matrix
    plt.clf()
    sn.heatmap(confusion_matrix(predictions_pandas["label"], predictions_pandas["prediction"], normalize='pred'), annot=True).set(xlabel="Predicted", ylabel = "Labels")
    cm_norm = stringfunctions.get_image()
    
    # precision, recall, fscore
    all_metrics = metrics.precision_recall_fscore_support(predictions_pandas["label"], predictions_pandas["prediction"])
    
    # Accuracy
    accuracy = metrics.accuracy_score(predictions_pandas["label"], predictions_pandas["prediction"])
    
    # precision
    precision_by_label = list(all_metrics[0])
    
    # recall, sensitivity or TP rate
    recall_by_label = list(all_metrics[1])
    
    # FP rate and TN Rate (we get them from cm)
    cmat = confusion_matrix(predictions_pandas["label"], predictions_pandas["prediction"])

    FP = cmat.sum(axis=0) - np.diag(cmat)  
    FN = cmat.sum(axis=1) - np.diag(cmat)
    TP = np.diag(cmat)
    TN = cmat.sum() - (FP + FN + TP)

    # Specificity or true negative rate
    TNR_by_label = TN/(TN+FP)
    
    # false positive rate
    FPR_by_label = FP/(FP+TN)
    
    # F1 measure or F1 Score
    F1_by_label = list(all_metrics[2])
    
    # get weighted measures
    precision_weighted = sum(precision_by_label)/len(precision_by_label)
    recall_weighted = sum(recall_by_label)/len(recall_by_label)
    TNR_weighted = sum(TNR_by_label)/len(TNR_by_label)
    FPR_weighted = sum(FPR_by_label)/len(FPR_by_label)
    F1_weighted = sum(F1_by_label)/len(F1_by_label)
    
    return num_folds, F1_mean, cm, cm_norm, accuracy, precision_by_label, recall_by_label, TNR_by_label, FPR_by_label, F1_by_label, precision_weighted, recall_weighted, TNR_weighted, FPR_weighted, F1_weighted