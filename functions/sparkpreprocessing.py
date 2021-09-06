# Pyspark required libraries
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def classification_preprocessing(variables_numeric, variables_categoric, target_variable, original_df, problem_type):
    
    # Source: https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
    
    # Get original variables, variables of the first dataframe
    original_variables = list(original_df.schema.names)
    
    # Categorical Columns preprocessing

    # Set up Pipeline stages
    stages = []
    for categoricalCol in variables_categoric:
        
        # A label indexer that maps a string column of labels to an ML column of label indices.
        # The name of the output cols will be the same as the input but with an Index at the end
        # This will be the step 1 of the Pipeline
        
        # Step 1 of the Pipeline: Convert categorical values to doubles.
        # Get its Index (with StringIndexer), each categorical variable is mapped to a float from 0.0, incresing by 1.0
        # Eg with 4 values in the variable: [0.0, 1.0, 2.0, 3.0]
        
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

        # One hot encoding
        # This will be the step 2 of the Pipeline
        
        # Step 2 of the Pipeline: Get its Vector(with OneHotEncoder), each categorical Indexed variable is maped to an array with this shape:
        # Do the one hot encoding. Convert each double value in a binary vector with a sparse representation.
        # The meaning of the sparse vector is:
        # First value: Length of the binary array
        # Second value: Index/Position of the value (1.0) in the binary array
        # Third value: Value (1.0)
        
        # Important about the shape of the sparse vectors:
        # A one-hot encoder that maps a column of category indices to a column of binary vectors, with at most a single one-value 
        # per row that indicates the input category index. For example with 5 categories, an input value of 2.0 would map to an output 
        # vector of [0.0, 0.0, 1.0, 0.0]. The last category is not included by default (configurable via OneHotEncoder!.dropLast 
        # because it makes the vector entries sum up to one, and hence linearly dependent. So an input value of 4.0 
        # maps to [0.0, 0.0, 0.0, 0.0]. Note that this is different from scikit-learn's OneHotEncoder, 
        # which keeps all categories. The output vectors are sparse
        
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])

        # Build the list that contains the stages of the pipeline (steps 1 and 2)
        stages += [stringIndexer, encoder]

    # Step 3 of the Pipeline (apply step 1 to target variable) - ONLY FOR CLASSIFICATION
    # The target variable does not get the OHEncoding, only the StringIndexer processing
    # Change the name of target variable for label
    if (problem_type != "Regression"):
        label_stringIdx = StringIndexer(inputCol = target_variable, outputCol = 'label')
        stages += [label_stringIdx]

    # Step 4 of the Pipeline: Select the columns and build a features vector, it will have the sparse vectors from Step 2 and the numerical columns
    # Columns for that will be the input of the features Vector, Vectorized categorical columns (The ones that got OHEncoding) 
    # and numeric columns, that stay the same way. 
    assemblerInputs = [c + "classVec" for c in variables_categoric] + variables_numeric

    # Merge the selected columns in an array in one single column (its name will be features)
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    # Pipeline: It consists in a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order. 
    # Pipeline stages defined previously
    
    # Build the pipeline and apply it
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(original_df)
    df = pipelineModel.transform(original_df)
    
    if (problem_type == "Regression"):
        df = df.withColumn("label", original_df[target_variable])
    
    # Get the required columns of the dataframe (original + label + features)
    selectedCols = ['label', 'features'] + original_variables
    df = df.select(selectedCols)

    return df