## Step1 Reading File
df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/rupanjalisingh10@gmail.com/spam.csv")
df1.show()
df1.printSchema()
df1.display()
df=df1.dropDuplicates()
df.display()
df2=df1.dropDuplicates()
display(df2)
df2.dropna
display(df2)
df_cleaned = df2.filter(df2._c2.isNotNull() & df2._c3.isNotNull() & df2._c4.isNotNull())
df_cleaned.display()
dfrename = df2.withColumnRenamed("v1", "source").withColumnRenamed("v2", "message")
display(dfrename)
dfrename.drop("_c2","_c3","_c4")
display(dfrename)
dfrename.printSchema()
### Data cleaning and Preprocessing

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import col, lower, regexp_replace

# Lowercase and remove punctuation
df3 = dfnew.withColumn("message", lower(col("message")))
df3 = dfnew.withColumn("message", regexp_replace(col("message"), "[^a-zA-Z\\s]", ""))
display(df3)

# Tokenization
tokenizer = Tokenizer(inputCol="message", outputCol="words")
wordsData = tokenizer.transform(dfnew)

# Stopword Removal
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)
dfrename.filter(dfrename.message.isNull()).show()
dfnew = dfrename.na.drop(subset=["message"])
display(dfnew)
### Feature Engineering
# TF-IDF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(filteredData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData = rescaledData.withColumn("source", rescaledData["source"].cast("int"))
rescaledData.printSchema()
### Model Building and feature engineering
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

# Create Spark session
spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()

# Sample data with labels (e.g., source column as labels)
data = [
    (0, "This is an example of TF-IDF.", "1"),
    (1, "TF-IDF computes term importance in documents.", "0"),
    (2, "This example demonstrates text feature extraction.", "1"),
    (3, "Another example for classification.", "0")
]
columns = ["id", "text", "source"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

# Compute term frequencies
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Compute TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Convert the 'source' column to integer
rescaledData = rescaledData.withColumn("source", rescaledData["source"].cast("int"))

# Split data into training and testing sets
train, test = rescaledData.randomSplit([0.8, 0.2], seed=1234)

# Train Logistic Regression Model
lr = LogisticRegression(featuresCol='features', labelCol='source')
lrModel = lr.fit(train)

# Evaluate model on test data
predictions = lrModel.transform(test)

# Show predictions
predictions.select("id", "text", "source", "prediction", "probability").show(truncate=False)

# Make predictions
predictions = lrModel.transform(test)

# Evaluate the model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="source", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
# Save the model
lrModel.save("/FileStore/models/sms_spam_model")
display(predictions.select("features", "source", "text", "id", "words"))
