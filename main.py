import argparse
import cleanup
import setup

from pyspark.sql import SparkSession

# These are the Spark ML library functions we'll be using
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel

# We'll also be using MLflow
import mlflow

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--user_name', help='Any unique text string that uniquely identifies the user')
parser.add_argument('--api_token', help='Databricks Personal Access Token')
parser.add_argument('--run_cleanup_at_eoj', help='y - run cleanup, any other value - do not run cleanup')
args = parser.parse_args()

# Get spark context
spark = SparkSession.builder.getOrCreate()

# Housekeeping - set up database tables, local and dbfs paths
# Notice the split() below...If I were coding this from scratch, I wouldn't build my return data this way...
# However, I'm trying to match my notebook version as closely as possible,
# and a notebook can only return a string.
setup_responses = setup.setup(args.user_name, spark, args.api_token).split()
local_data_path, dbfs_data_path, database_name = setup_responses

print(f"Path to be used for Local Files: {local_data_path}")
print(f"Path to be used for DBFS Files: {dbfs_data_path}")
print(f"Database Name: {database_name}")

# Set default Delta Lake database name
# Let's set the default database name so we don't have to specify it on every query
spark.sql(f"USE {database_name}")

# Rename "price" column to "label"
# Read the data into a dataframe and rename the "price" column to "label" to indicate that this is the
# value we want our model to predict
df_input = spark.sql("SELECT * FROM house_data")
df_input = df_input.withColumnRenamed("price", "label")

# We'll create a View so we can use SQL against df_input whenever we want
df_input.createOrReplaceTempView("vw_input")

# Explore the data to find the most useful features
# Let's narrow down our data to the features we actually want to use to build the model
# We'll also rename our "price" column to "label" to show that this is what we are trying to predict

# Some best practices for feature selection:
# - Eliminate dependent variables (for example, if there were a "sales tax" column, that would be directly dependent
# on the price... which we won't know at prediction time)
# - Make sure each variable has a sufficiently wide range of values (research min, max, avg, and look for a wide
#   range of actual values... high cardinality - use DESCRIBE to show)
# - Look for and choose variables that are good potential predictors (we'll use corr() method to find this)
# - Look for and eliminate "duplicate" variables (those with high correlation to each other).  These can skew results.

# Here we use describe() to examine our data to see which columns have a sufficiently wide range of values
# Some things to note...
#  - id field is not useful
#  - date range is roughly a year, which is good because we don't have to worry about the way price changes over
#    long periods of time
#  - waterfront is either 0 or 1 (so we'll need to encode it as categorical data)
#  - view means how many times the house has been views (so it's not useful for us)
#  - grade is the overall grade given to the property by the King County grading system (it's a number between 1
#    and 13).  We can't assume that higher is better, so we'll have to encode it as categorical data
#  - yr_built looks like a good choice.  However, yr_renovated is questionable.  There are lots of 0s in it.
#    If we do use it, we might want to treat is as a boolean (renovated or not renovated)
#  - zip code will definitely have to be treated as a categorical, because higher is not necessarily better or worse.
#  - lat and long are not useful, and at any rate there is very little variation because all the homes are in
#    King County
df_explorer = df_input
print("Below we describe dataframe df_explorer")
print(df_explorer.describe())

# Find the correlation between each column and the label (price)
# Next, let's check the actual correlation between each column and the price (which is now named "label").
# We'll get a number between 1 and -1.  Each of those extremes represents high (positive or negative) correlation.
# Numbers near 0 represent low correlation.  We don't want to use those columns

# For the moment, we'll ignore Categorical columns.  We'll deal with them later.
df_correlation_explorer = df_explorer.drop("id", "date", "waterfront", "view", "condition", "grade", "yr_renovated",
                                           "zipcode", "lat", "long")

for col in df_correlation_explorer.columns:
    print(f"The correlation between label (price) and {col} is {df_correlation_explorer.stat.corr('label', col)}")

# Eliminate columns with low correlation to price
# First we'll get rid of the variables we found above that have low correlation to price.
# We'll also get rid of sqft_living15, because we have seen that it is very close to sqft_living

df_correlation_explorer = df_correlation_explorer.drop("sqft_lot", "sqft_lot15", "yr_built", "sqft_living15")

# Next we'll look for (and potentially eliminate) variables that are highly correlated to each other - "duplicate"
# variables that can skew the model
# We'll compare the correlation of each column to each other column
# If we find two columns that are highly correlated with each other, we'll KEEP the one with the higher correlation
# to price.
# Keep in mind, though, that we have to use common sense.  For example, bathrooms and bedrooms also have pretty
# high correlation for sqft_living, but I'm going to keep them because it makes sense that these would affect price
# (of course, I could always prove my theories via actual evaluation of models).
for col1 in df_correlation_explorer.columns :
  for col2 in df_correlation_explorer.columns :
    print(f"The correlation between {col1} and {col2} is {df_correlation_explorer.stat.corr(col1, col2)}")

# Calculate a useful feature column
# We saw above that sqft_living, sqft_above, and sqft_basement are relatively correlated.
# If it's usually bad to have two variables that are  highly correlated, it's even worse to have three!
# We're going to compute a single column that should provide a better variable: percentage of square footage
# above ground.
df_input = spark.sql(
    """
    SELECT 
      *,
      sqft_above / sqft_living AS sqft_above_percentage
    FROM vw_input
    """
)

# Now we'll refresh the temporary view of our raw input so we can run SQL against it
df_input.createOrReplaceTempView("vw_input")

# Split input into training/test and holdout chunks
# Split the input into two dataframes to be used for (1) training-and-testing and (2) holdout data
# We'll be using ParamGrid to test many hyperparameter combinations, which is why we need a training-and-testing
# chunk for Spark because Spark will figure out the best model for us),
# and another holdout chunk for us to test independently.
(df_input_training_and_test, df_input_holdout) = df_input.randomSplit([0.7, 0.3], seed=100)

# Create indexes for Categorical columns
# Now let's deal with Categorical columns
# Categoricals are fields where higher values do not indicate superiority, and lower values do not indicate inferiority.
# For example, zip codes are numeric, but a higher zip code is not "bigger than" or "better than" a lower zip code.
# Another example. We could change a "gender" column to 1 for male, 2 for female, and 3 for unknown.
# However, this of course does NOT mean that females are "more than" males.
# To avoid this "more than" problem, we use "one-hot encoding" to create "vectors" to encode this data.
# There is a position in the vector for each data value.
# For each row, a 1 is placed in the vector index position that represents the value we want to encode.
# All other values in the vector are set to 0.  That's why we call it "one-hot" encoding.

# There are two steps to this process of dealing with Categoricals.
#  1. Indexing: assigning a numerical value to each data value
#  2. Encoding: creating the vector.

# First we'll index (NOTE that StringIndexer works on numeric data as well)
conditionIndexer = StringIndexer(inputCol="condition", outputCol="condition_index")

gradeIndexer = StringIndexer(inputCol="grade", outputCol="grade_index")

zipcodeIndexer = StringIndexer(inputCol="zipcode", outputCol="zipcode_index")

# Encode the indexed Categorical columns into Vectors
# Now we'll encode into vectors

# Now we'll tranform the indexed values into a vector
encoder = OneHotEncoder()
encoder.setInputCols(["condition_index", "grade_index", "zipcode_index"])\
    .setOutputCols(["condition_vector", "grade_vector", "zipcode_vector"])

# Transform all Features into a single Vector
# Transform the features into a Spark ML Vector

# Let's define our vector with only the features we actually want to use to build the model
# We'll ignore the columns above that are highly correlated to one another.

# Note that waterfront is treated as a boolean, so we didn't have to encode it.
# We can just add it to the vector assembler.
assembler = VectorAssembler(
    inputCols=["bedrooms", "bathrooms", "sqft_living", "sqft_above_percentage", "floors", "condition_vector", "grade_vector", "zipcode_vector", "waterfront"],
    outputCol="features")

# Build a Grid of Hyperparameters to test
# Here we build a Grid of hyperparameters so we can test all permutations

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
lr = LinearRegression()

paramGridBuilder = ParamGridBuilder()

paramGrid = paramGridBuilder\
  .addGrid(lr.regParam, [0.01, 0.1, 0.5])\
  .addGrid(lr.elasticNetParam, [0, 0.5, 1])\
  .build()

# Split data into Training and Testing chunks, and prepare to build model
# In order to test many hyperparameter combinations and choose the best-performing model, we use a
# TrainValidationSplit object.
# TrainValidationSplit requires us to supply 4 parameters:
#  1. An estimator.  This is the model builder we will use.  In our case, it is a LinearRegression object
#  2. An evaluator.  This tells how we want to evaluate results to determine which model is best.
#  3. An estimatorParamMaps.  This is the ParamGrid object with all the hyperparameter values
#  4. A tranRatio.  This is a decimal fraction between 0 and 1 that tells us which part of the training-and-test
#     data to use for training.  The remainder is used for the evaluator's own testing to determine best model
tvs = TrainValidationSplit(
  estimator=lr,
  estimatorParamMaps=paramGrid,
  evaluator=RegressionEvaluator(), # This defaults to using best RMSE, can be set to use R2
  trainRatio=0.8
)

# Test Hyperparameter values and determine the best model.  Persist the best model.
# Here we create a pipeline that will execute the objects we created above,
# which include indexers, one-hot encoder, vector assembler, param grid, and train-validation split

# Note that we have included the train-validation split in the parameters below, but not the
# linear regression object or the param grid. These are already parameters in the train-validation split.

# Note the 6 parameters that make up the pipeline stages.

# By using a pipeline, we avoid running fit() and transform() methods on each of the individual components,
# which we defined in the cells above.
# This can be advantageous in some development situations.  For example, if we want to try multiple variations
# of a vector assembler, we simply code multiple versions, and then plug our current choice into the pipeline.
# We can than easily switch back and forth as we test.

# However, pipelines can also have some disadvantages.
# First, pipelines make it a bit more complex to check accuracy and parameters of a model, because we have to
# extract from the pipeline before we can extract the model.summary information.  We'll see this in the cell below.
# Second, pipelines can add complexity when we use them with multiple different data sets.  We'll see this when
# we run against our holdout data two cells below.  As you will see, we have to drop the prediction column before
# we run, which makes our pipeline inefficient.

# Spark ML will return the best model.  To determine "best," we let Spark ML use its default measure for linear
# regression, which is Root Mean Squared Error (RMSE)
# We'll use MLflow to log the best model, so others can retrieve and use it.
with mlflow.start_run() as run:
    pipeline = Pipeline()
    pipeline.setStages([conditionIndexer, gradeIndexer, zipcodeIndexer, encoder, assembler, tvs])
    pipelineModel = pipeline.fit(df_input_training_and_test)

    # Log the best model
    mlflow.spark.log_model(pipelineModel, "house-price-pipelineModel")

    best_run = run.info
    print(f"Best run: {best_run}")

# Print some interesting data about the best model
# Check model accuracy and chosen parameters

# To get the model, you need to access the TrainValidationSplit object in the pipeline, which is the 6th param
# (see the constructor above)
model = pipelineModel.stages[5]

# Now that you have model, you can access bestModel.summary to get all the values below.
print(f"The model's R2 (closer to 1 is better): {model.bestModel.summary.r2}")
print(f"The model's RMSE (smaller is better): {model.bestModel.summary.rootMeanSquaredError}")
print(f"The model's reg param: {model.bestModel.getRegParam()}")
print(f"The model's elastic net param: {model.bestModel.getElasticNetParam()}")
print(f"The model's intercept: {model.bestModel.intercept}")
print(f"The model's coefficients: {model.bestModel.coefficients}")

# Prepare to verify the model against the Holdout data
# Check the model's accuracy against the holdout data
# Our hope is that the model's accuracy against the test data (above) is close to its accuracy against the holdout
# data (here).
# If this is the case, then we can feel confident that our model will be generally useful, and not just "overtrained"
# against the test data.

# Keep in mind that our holdout data has not gone through the pipeline, so it does not have the index columns or
# vectors or feature column.
# We'll need to run it through the pipeline to get these.
# Then (and this is kludgey) we have to drop the prediction column.  So this is not very efficient to run, but at
# least the code is briefer than if we ran all the pipeline steps individually.

df_holdout_prepared = pipelineModel.transform(df_input_holdout).drop("prediction")

# To get the model, you need to access the TrainValidationSplit object in the pipeline, which is the 6th param
# (see the constructor above)
model = pipelineModel.stages[5]

# Now that you have model, you can access bestModel.summary to get all the values below.
print(f"The test data R2 (closer to 1 is better): {model.bestModel.evaluate(df_holdout_prepared).r2}")
print(f"The test data RMSE (smaller is better): {model.bestModel.evaluate(df_holdout_prepared).rootMeanSquaredError}")

# Run the model against the Holdout data and predict prices
# Make predictions on test data. model is the model with combination of parameters
# that performed best.

# The predictions column is the rightmost column in the displayed result.
df_predictions = model.transform(df_holdout_prepared)\
  .select("*")

print("Below we print the first 20 rows of predictions against holdout data")
df_predictions.show()

# Section 2: Load and execute the model
# Now we'll "change hats" and pretend we are a different developer. We'll load the model created and persisted above,
# and run it against a new batch of data.

# Recreate the calculated column on a new batch of data
# Load in our unlabeled data, and concoct the sqft_above_percentage column just like we did when developing the model
df_in = spark.sql(
    """
    SELECT 
      *,
      sqft_above / sqft_living AS sqft_above_percentage
    FROM house_data_unlabeled
    """
)

# Retrieve interesting information about the model we want to use
# In a real-world situation, we would need to be informed about the Experiment ID.  We would get this information
# from the MLflow Server.
# However, since we are in a single notebook, we'll just print the information about the Experiment this notebook
# is using.
best_run = run.info
print(f"Best run: {best_run}")

# Load the best model
# Now we'll load the pipeline that was created and logged above
# We can use the Artifact ID from the Experiment's run.info
logged_model = f'{run.info.artifact_uri}/house-price-pipelineModel'

# Load model
model = mlflow.spark.load_model(logged_model)

# Use the model to predict prices on the new data
# Make the predictions on the input data
df_with_predictions = model.transform(df_in)
print("Below we print the first 20 rows of predictions against the unlabeled data")
df_with_predictions.show()

# Clean up database and files
if args.run_cleanup_at_eoj.lower() == 'y':
    print(cleanup.cleanup(args.user_name, spark))
