##################################################
## Project: Lesson 2 homework
## Script purpose: Main Executable R Script
## Date: 02/10/2019
## Author: Krzysztof Leszek
##################################################

# Packages required  for execution
# install.packages("class")  
# install.packages("caret") 
# install.packages("gmodels") 

# Additional Packages/Libraires needed for execution
library(class) # for k-NN alghoritm
library(caret) # for dummyVars and one-hot encoding of categorical variables
library(gmodels) # for model performance evaluation (CrossTable)

# Config Items
working_directory <- "/_My_Local/Projects/COMPSCI X460/Lesson_2"
file_name <- "EmployeeData.csv"
setwd(working_directory)  # Set working directory

## Fetch Source Data
source_data <- read.csv(file_name, stringsAsFactors = FALSE)

## Prepare Data
# Drop variables having either a single value for all records
# or unique value for all records as they will not 
# contribute to KNN prediction
dropMininglessVariables <- function(data_set) {
  columns <- names(data_set)
  prep_data_set <- data_set
  
  for (column in columns) {
    cat("\n Processing column ", column)
    
    # Drop if column has only one value for all records 
    if (dim(table(data_set[,column])) == 1) {
      cat(" => Dropping column ", column, " (column contains single value for all records)")
      prep_data_set <- prep_data_set[ , !(names(prep_data_set) %in% column)]
      
    } else if (dim(table(data_set[,column])) == nrow(data_set)) {
      # Drop if column has different values for all records 
      cat(" => Dropping ", column, " (column contains unique values for all records)")
      prep_data_set <- prep_data_set[ , !(names(prep_data_set) %in% column)]
    } else {
      cat(" => column OK")
    }
  }
  
  return(prep_data_set)
}

# Drop meaningless variables
source_data <- dropMininglessVariables(source_data)
str(source_data)

# Code outcome feature as factor (required by k-NN)
source_data$IncomeLevel <- factor(source_data$IncomeLevel)

# Code other variables that are categotical to factor
#table(source_data$JobLevel)
source_data$Education <- factor(source_data$Education)
source_data$EnvironmentSatisfaction  <- factor(source_data$EnvironmentSatisfaction)
source_data$JobLevel <- factor(source_data$JobLevel)
source_data$StockOptionLevel <- factor(source_data$StockOptionLevel)

outcomeName <- c("IncomeLevel")
outcomes <- source_data[ , outcomeName]

# One-hot encode all other categotical features 
dmy <- dummyVars(" ~ .", data = source_data[ , !names(source_data) %in% outcomeName])
predictors <- data.frame(predict(dmy, newdata = source_data))
str(predictors)

# Visualize data
# Graph 1
par(mfrow=c(3,2))
par(mar=c(2.5,2.5,2.5,2.5), font.axis = 2, font.lab = 2)
#graphics.off()
hist(predictors$Age, main = "Histogram of Age", xlab = "Age")
hist(predictors$DailyRate, main = "Histogram of DailyRate", xlab = "DailyRate")
hist(predictors$DistanceFromHome, main = "Histogram of DistFromHome", xlab = "DistFromHome")
#hist(predictors$Education, main = "Histogram of Education", xlab = "Education")
#hist(predictors$EnvironmentSatisfaction, main = "Histogram of EnvSatisfaction", xlab = "EnvSatisfaction")
hist(predictors$HourlyRate, main = "Histogram of HourlyRate", xlab = "HourlyRate")
hist(predictors$JobInvolvement, main = "Histogram of JobInvolvement", xlab = "JobInvolvement")
hist(predictors$MonthlyIncome, main = "Histogram of MonthlyIncome", xlab = "MonthlyIncome")
# Graph 2
par(mfrow=c(4,2))
par(mar=c(2.5,2.5,2.5,2.5), font.axis = 2, font.lab = 2)
hist(predictors$MonthlyRate, main = "Histogram of MonthlyRate", xlab = "MonthlyRate")
hist(predictors$DailyRate, main = "Histogram of DailyRate", xlab = "DailyRate")
hist(predictors$NumCompaniesWorked, main = "Histogram of NumCompaniesWorked", xlab = "NumCompaniesWorked")
hist(predictors$PercentSalaryHike, main = "Histogram of PercentSalaryHike", xlab = "PercentSalaryHike")
hist(predictors$PerformanceRating, main = "Histogram of PerformanceRating", xlab = "PerformanceRating")
hist(predictors$RelationshipSatisfaction, main = "Histogram of RelSatisfaction", xlab = "RelSatisfaction")
#hist(predictors$StockOptionLevel, main = "Histogram of StockOptionLevel", xlab = "StockOptionLevel")
hist(predictors$TotalWorkingYears, main = "Histogram of TotalWorkingYears", xlab = "TotalWorkingYears")
# Graph 3
par(mfrow=c(3,2))
par(mar=c(2.5,2.5,2.5,2.5), font.axis = 2, font.lab = 2)
hist(predictors$TrainingTimesLastYear, main = "Histogram of TrainingTimesLastYear", xlab = "TrainingTimesLastYear")
hist(predictors$WorkLifeBalance, main = "Histogram of WorkLifeBalance", xlab = "WorkLifeBalance")
hist(predictors$YearsAtCompany, main = "Histogram of YearsAtCompany", xlab = "YearsAtCompany")
hist(predictors$YearsInCurrentRole, main = "Histogram of YearsInCurrentRole", xlab = "YearsInCurrentRole")
hist(predictors$YearsSinceLastPromotion, main = "Histogram of YearsSinceLastPromotion", xlab = "YearsSinceLastPromotion")
hist(predictors$YearsWithCurrManager, main = "Histogram of YearsWithCurrManager", xlab = "YearsWithCurrManager")

# Apply nrmalization on other numeric features
features_to_normalize <- c("Age",
                           "DailyRate",
                           "DistanceFromHome",
                           #"Education",
                           #"EnvironmentSatisfaction",
                           "HourlyRate",
                           "JobInvolvement",
                           #"JobLevel",
                           "JobSatisfaction",
                           "MonthlyIncome",
                           "MonthlyRate",
                           "NumCompaniesWorked",
                           "PercentSalaryHike",
                           "PerformanceRating",
                           "RelationshipSatisfaction",
                           #"StockOptionLevel",
                           "TotalWorkingYears",
                           "TrainingTimesLastYear",
                           "WorkLifeBalance",
                           "YearsAtCompany",
                           "YearsInCurrentRole",
                           "YearsSinceLastPromotion",
                           "YearsWithCurrManager"
)

# Function performing normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization and prepare predictor dataset
# Method 1 (min/max)
predictors_normalized <- as.data.frame(lapply(predictors[features_to_normalize], normalize))
# Method 1 (z-score)
#predictors_normalized <- as.data.frame(scale(predictors[features_to_normalize]))
predictors[ , features_to_normalize] <- predictors_normalized

# Append outcome
source_data_prepared <-cbind(predictors, IncomeLevel=outcomes)

# Define training and testing set with labels
set.seed(1234)   
indicator <- sample(2, nrow(source_data_prepared), replace = T, prob = c(0.80, 0.20))  # 80/20 for training

# Training dataset
training_dataset <- source_data_prepared[indicator == 1,]
training_dataset_outcome <- training_dataset$IncomeLevel   # store outcome feature as factor vector (required by k-NN)
training_dataset <- training_dataset[ , !(names(training_dataset) %in% "IncomeLevel")]
# Print % of outcomes in training set
round(prop.table(table(training_dataset_outcome)) * 100, digits = 1)

# Testing dataset
testing_dataset <- source_data_prepared[indicator == 2, ]
testing_dataset_outcome <- testing_dataset$IncomeLevel # store outcome feature as factor vector (required by k-NN)
testing_dataset <- testing_dataset[ , !(names(testing_dataset) %in% "IncomeLevel")]
# Print % of outcomes in testing set
round(prop.table(table(testing_dataset_outcome)) * 100, digits = 1)

# Check if dimentions are the same
ncol(training_dataset)
ncol(testing_dataset)

# Train the model on the data
testing_dataset_preditions <- knn(train = training_dataset, test = testing_dataset,
                                  cl = training_dataset_outcome, k = 35)

# Evaluate model performance
CrossTable(x = testing_dataset_preditions, y = testing_dataset_outcome, prop.chisq = FALSE)

# Definitions
# "High" is a positive class.
# "Low" is a negative class.
