#THE FOLLOWING IS AN R SCRIPT CONTAINING ALL THE CODE USED IN THE CREATION OF THIS MODEL

#THE SCRIPT CONTAINS A SERIES OF CODE CHUNKS, EACH CHUNK HAS A NAME THAT IS CONSISTENT WITH ITS NAME IN THE .Rmd FILE

#THE SCRIPT IS DIVIDED INTO 3 SECTIONS, NAMELY:

#1 - DATA PREPARATION

#2 - MODEL FITTING - INCLUDES 5 SUB-SECTIONS, NAMELY:
###I - SPLITTING DATA INTO TRAINING AND TEST SETS
###II - LOGISTIC REGRESSION MODEL
###III - KNN MODEL
###IV - DECISION TREE MODEL
###V - RANDOM FOREST MODEL

#3 - PLOTS AND TABLES MADE - INCLUDES 3 SUBSECTIONS, NAMELY:
###I - PLOTS AND TABLES USED TO VISUALIZE DATA STRUCTURE
###II - PLOTS AND TABLES USED TO VISUALIZE MODEL PERFORMANCE - INCLUDES A SEPARATE SECTION FOR EACH MODEL
###III - PLOTS AND TABLES USED TO VISUALIZE RESULTS


#################
#DATA PREPARATION
#################

####SECTION START


##CHUNK NAME: setup

#CHUNK START
#Checking if required packages are installed. If these are not available, they are automatically installed with the following code
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(ggcorrplot)) install.packages("ggcorrplot")
if(!require(knitr)) install.packages("knitr")
if(!require(rpart.plot)) install.packages("rpart.plot")

#Loading required libraries
library(tidyverse)
library(caret)
library(gridExtra)
library(ggcorrplot)
library(knitr)
library(rpart.plot)
#CHUNK END

##CHUNK NAME: data_prep

#CHUNK START
#Creating a temporary directory to store the downloaded file
temp <- tempfile()
#Storing the URL of the data file in a variable
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
#Downloading the file and storing it in the temporary directory
download.file(url, destfile = temp)
#Reading the file into a data.frame in R
banknote_tab <- read.table(temp, header = FALSE, sep = ",")
#Deleting temporary variables
rm(temp, url)
#CHUNK END

##CHUNK NAME: cleaning1

#CHUNK START
#Defining column names
names(banknote_tab) <- c("Variance", "Skewness", "Kurtosis", "Entropy", "Class")
#CHUNK END

##CHUNK NAME: cleaning2

#CHUNK START
#Redefining 'Class' as a factor
banknote_tab <- banknote_tab %>% mutate(Class = factor(Class))
#CHUNK END

####SECTION END


##############
#MODEL FITTING
##############

####SECTION START

###SUBSECTION I - SPLITTING DATA INTO TRAINING AND TEST SETS
###SUBSECTION START

##CHUNK NAME: subset

#CHUNK START
#Setting the seed to 1 in order to enable reproduction of datasets
set.seed(1, sample.kind = "Rounding")
#Defining an Index to subset 20% of the data
validation_index <- createDataPartition(banknote_tab$Class, times = 1, p = 0.2, list = FALSE)
#Defining the training set, 'edx', as 80% of the data
edx <- banknote_tab %>% slice(-validation_index)
#Defining the test set, 'validation', as 20% of the data
validation <- banknote_tab %>% slice(validation_index)
#CHUNK END

##CHUNK NAME: caret_prep

#CHUNK START
#Excluding the Class (outcome) column and converting to matrix
edx_features <- as.matrix(edx %>% select(-Class))
#Extracting the Class column of the training set
edx_outcome <- edx$Class
#CHUNK END

###SUBSECTION END

###SUBSECTION II - LOGISTIC REGRESSION MODEL
###SUBSECTION START

##CHUNK NAME: glm_model

#CHUNK START
#Creating a logsitic regression model using the glm function
#The argument 'family = binomial(link = "logit")'
#Specifies that a logistic regression model is to be created
glm_model <- glm(Class ~ ., data = edx, family = binomial(link = "logit"))
#CHUNK END

##CHUNK NAME: glm_probs

#CHUNK START
#Computing the probability of each banknote in the validation set being genuine
probs <- predict(glm_model, newdata = validation, type = "response")
#CHUNK END

##CHUNK NAME: glm_preds

#CHUNK START
#Predicting the class of each banknote in the validation set
#Banknote is predicted to be genuine (1) if probability is greater than 0.5
#Otherwise, it is predicted to be a specimen (0)
glm_preds <- ifelse(probs > 0.5, 1, 0) %>% factor()
#CHUNK END

###SUBSECTION END

###SUBSECTION III - KNN MODEL
###SUBSECTION START

##CHUNK NAME: knn_model

#CHUNK START
#Creating and training the KNN Model
#The 'tuneGrid' argument is used to define a set of values for K
#Which is a tuning parameter and needs to be optimized
knn_model <- train(edx_features, edx_outcome,
                   method = "knn",
                   tuneGrid = data.frame(k = seq(2, 10, 2)))
#CHUNK END

##CHUNK NAME: knn_preds

#CHUNK START
#Making predictions on the validation set with the KNN Model
knn_preds <- predict(knn_model, newdata = validation)
#CHUNK END

###SUBSECTION END

###SUBSECTION IV - DECISION TREE MODEL
###SUBSECTION START

##CHUNK NAME: rpart_model

#CHUNK START
#Creating and training the Decision Tree Model
#The 'tuneGrid' argument is used to define a set of value to test as CP
#Which is a tuning parameter that needs to be optimized
rpart_model <- train(edx_features, edx_outcome, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.01, 0.05, 0.005)))
#CHUNK END

##CHUNK NAME: rpart_preds

#CHUNK START
#Making predictions on the validation set using the decision tree model
rpart_preds <- predict(rpart_model, newdata = validation)
#CHUNK END

###SUBSECTION END

###SUBSECTION V - RANDOM FOREST MODEL
###SUBSECTION START

##CHUNK NAME: rf_model

#CHUNK START
#Creating and training the Random Forest Model
#The argument 'tuneGrid' is used to define a set of values
#Which are to be tested for the parameter mtry
rf_model <- train(edx_features, edx_outcome, 
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(1, 4, 1)))
#CHUNK END

#CHUNK NAME: rf_preds

#CHUNK START
#Making predictions on the validation set using the Random Forest Model
rf_preds <- predict(rf_model, newdata = validation)
#CHUNK END

###SUBSECTION END

####SECTION END


######################
#PLOTS AND TABLES MADE
######################

####SECTION START

###SUBSECTION I - PLOTS AND TABLES USED TO VISUALIZE DATA STRUCTURE
###SUBSECTION START

##CHUNK NAME: struct

#CHUNK START
#Using the str() function to get an overview of the structure of the dataset
str(banknote_tab)
#CHUNK END

##CHUNK NAME: testing_NAs

#CHUNK START
#Searching for NAs
sum(is.na(banknote_tab))
#CHUNK END

##CHUNK NAME: summary

#CHUNK START
#Applying the summary() function on the dataset to get an overview of the distributions of each feature
summary(banknote_tab)
#CHUNK END

##CHUNK NAME: density_plots

#CHUNK START
#Creating a smooth density plot to visualize the distribution of the variance of each banknote image in the dataset
Variance <- banknote_tab %>%
  ggplot(aes(Variance)) +
  geom_density(size = 1, color = "black", fill = "red", alpha = 0.2) +
  ylab("Density")
#Creating a smooth density plot to visualize the distribution of the entropy of each banknote image in the dataset
Entropy <- banknote_tab %>%
  ggplot(aes(Entropy)) +
  geom_density(size = 1, color = "black", fill = "red", alpha = 0.2) +
  ylab("Density")
#Creating a smooth density plot to visualize the distribution of the kurtosis of each banknote image in the dataset
Kurtosis <- banknote_tab %>%
  ggplot(aes(Kurtosis)) +
  geom_density(size = 1, color = "black", fill = "red", alpha = 0.2) +
  ylab("Density")
#Creating a smooth density plot to visualize the distribution of the skewness of each banknote image in the dataset
Skewness <- banknote_tab %>%
  ggplot(aes(Skewness)) +
  geom_density(size = 1, color = "black", fill = "red", alpha = 0.2) +
  ylab("Density")
#Displaying all 4 of the above defined plots in a square-like fashion
grid.arrange(Variance, Skewness, Kurtosis, Entropy, nrow = 2)
#CHUNK END

##CHUNK NAME: corrplot

#CHUNK START
#Computing a correlation matrix of the 4 features present in the dataset
cors <- cor(as.matrix(banknote_tab %>% select(-Class)))
#Visualizing the computed correlations using ggplot
#The argument 'hc.order = TRUE' is used to specify that the matrix is to be reordered using heirarchical clustering
ggcorrplot(cors, method = "square",
           legend.title = "Correlation",
           col = c("red", "black", "blue"),
           hc.order = TRUE)
#CHUNK END

###SUBSECTION END

###SUBSECTION II - PLOTS AND TABLES USED TO VISUALIZE MODEL PERFORMANCE
###SUBSECTION START

##########################
#LOGISTIC REGRESSION MODEL
##########################

##CHUNK NAME: prob_plot

#CHUNK START
#Visualizing the probabilites computed by the logistc regression model for each banknote image in the validation set through a smooth density plot made using ggplot2
data.frame(p = probs) %>%
  ggplot(aes(p)) +
  geom_density(size = 1, col = "black", fill = "red", alpha = 0.2) +
  xlab("Probability") + ylab("Density")
#CHUNK END

##########
#KNN MODEL
##########

##CHUNK NAME: knn_plot

#CHUNK START
#Visualizing the variation of cross-validation accuracy with increasing no. of neighbors K through a line plot made using ggplot2
knn_model$results %>%
  ggplot(aes(k, Accuracy)) +
  geom_line(color = "red", size = 1) +
  geom_point(shape = 1, size = 5, color = "black") +
  geom_point(shape = 16, color = "black") +
  xlab("No. of Neighbors (K)")
#CHUNK END

####################
#DECISION TREE MODEL
####################

##CHUNK NAME: rpart_plot

#CHUNK START
#Visualizing the variation of cross-validation accuracy with increasing value of the complexity parameter CP through a line plot made using ggplot2
rpart_model$results %>%
  ggplot(aes(cp, Accuracy)) +
  geom_line(color = "red", size = 1) +
  geom_point(color = "black", size = 5, shape = 1) +
  geom_point(color = "black", shape = 16) +
  xlab("Complexity Parameter")
#CHUNK END

##CHUNK NAME: rpart_tree

#CHUNK START
#Outputting the decision tree used by the final model to make predictions on the validation set using the rpart.plot package
rpart.plot(rpart_model$finalModel, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
#CHUNK END

####################
#RANDOM FOREST MODEL
####################

##CHUNK NAME: rf_acc_plot

#CHUNK START
#Visualizing the variation of cross-validation accuracy with increasing value of the tuning parameter mtry through a line plot made using ggplot2
rf_model$results %>%
  ggplot(aes(mtry, Accuracy)) +
  geom_line(color = "red", size = 1) +
  geom_point(shape = 1, color = "black", size = 5) +
  geom_point(shape = 16, color = "black") +
  xlab("No. of features considered at each node (mtry)")
#CHUNK END

##CHUNK NAME: rf_err_plot

#CHUNK START
#Visualizing the variation of OOB error rate with increasing value of the tuning parameter ntree through a line plot made using ggplot2
data.frame(ntree = seq(1, 500, 1),
           OOB = rf_model$finalModel$err.rate[,1]) %>%
  ggplot(aes(ntree, OOB)) +
  geom_line(color = "red", size = 1) +
  xlab("Number of Trees (ntree)") + ylab("OOB Error Rate")
#CHUNK END

######################
#SUMMARY OF ALL MODELS
######################

##CHUNK NAME: model_table

#CHUNK START
#Creating a tibble containing the tuning parameters (if any) of each model and their respective values set after opitmization
model_table <- tibble(Model = c("Logistic Regression Model", "K-Nearest-Neighbors (KNN) Model", "Decision Tree Model", "Random Forest Model"),
                      Tuning_Parameters = c("None", "No. of Neighbors (K)", "Complexity Parameter (CP)", "No. of features considered at each node (mtry)"),
                      Value = c("-", knn_model$bestTune$k, rpart_model$bestTune$cp, rf_model$bestTune$mtry))
#Using the knitr package to display the tibble in a neat manner
#The 'align = rep("c", 3)' argument is used to create a vector that specifies the alignment of each colum. In this case, as all columns are to be center aligned, the vector contains 3 occurences of "c", short of "center".
kable(model_table,
      align = rep("c", 3),
      col.names = c("Model", "Tuning Parameters", "Optimized Value"))
#CHUNK END

###SUBSECTION END

###SUBSECTION III - PLOTS AND TABLES USED TO VISUALIZE RESULTS
###SUBSECTION START

##CHUNK NAME: preds_barplot

#CHUNK START
#Creating a data.frame that contains the number of occurences of each class in the predictions made by each model along with the actual occurences of each class in the validation set
model_preds <- data.frame(model = c(rep("Logistic Regression", 2), 
                                    rep("KNN Model", 2),
                                    rep("Decision Tree", 2),
                                    rep("Random Forest", 2),
                                    rep("Actual Class", 2)),
                          class = rep(c("0", "1"), 5),
                          total = c(sum(glm_preds == 0), sum(glm_preds == 1),
                                    sum(knn_preds == 0), sum(knn_preds == 1),
                                    sum(rpart_preds == 0), sum(rpart_preds == 1),
                                    sum(rf_preds == 0), sum(rf_preds == 1),
                                    sum(validation == 0), sum(validation == 1)))
#Using the above defined data.frame to generate a barplot visualizing the occurence of each class in the predictions made by each model and comparing them with the occurence of each class in the validation set
#The 'coord_cartesian()' function is used to define a custom range for the y-axis
model_preds %>% ggplot(aes(x = model, y = total, fill = factor(class))) + 
geom_bar(stat = "identity", position = "dodge", width = 0.5, color = "black") + 
scale_fill_manual(values = c("#FF000050", "#0000FF50")) +
xlab("Model Type") + ylab("Total Count") +
coord_cartesian(ylim = c(100, 165)) +
theme(axis.text.x = element_text(angle = 45, hjust = 1)) + labs(fill = "Class")
#CHUNK END
  
##CHUNK NAME: results_barplot
  
#CHUNK START
#Computing the confusion matrix of each model
glm_res <- confusionMatrix(glm_preds, validation$Class)
knn_res <- confusionMatrix(knn_preds, validation$Class)
rpart_res <- confusionMatrix(rpart_preds, validation$Class)
rf_res <- confusionMatrix(rf_preds, validation$Class)

#Extracting the accuracy of each model from their repsective confusion matrices
glm_acc <- glm_res$overall["Accuracy"]
knn_acc <- knn_res$overall["Accuracy"]
rpart_acc <- rpart_res$overall["Accuracy"]
rf_acc <- rf_res$overall["Accuracy"]

#Extracting the sensitivity of each model from their repsective confusion matrices
glm_sen <- glm_res$byClass["Sensitivity"]
knn_sen <- knn_res$byClass["Sensitivity"]
rpart_sen <- rpart_res$byClass["Sensitivity"]
rf_sen <- rf_res$byClass["Sensitivity"]

#Extracting the specificity of each model from their repsective confusion matrices
glm_spec <- glm_res$byClass["Specificity"]
knn_spec <- knn_res$byClass["Specificity"]
rpart_spec <- rpart_res$byClass["Specificity"]
rf_spec <- rf_res$byClass["Specificity"]

#Compiling the accuracy, sensitivity and specificity of each model in a data.frame
results <- data.frame(Model = as.factor(c("Logistic Regression Model",
                                          "KNN Model",
                                          "Decision Tree Model",
                                          "Random Forest Model")), 
                      Accuracy = c(glm_acc, knn_acc, rpart_acc, rf_acc),
                      Specificity = c(glm_spec, knn_spec, rpart_spec, rf_spec),
                      Sensitivity = c(glm_sen, knn_sen, rpart_sen, rf_sen))

#Creating a bar plot visualizing the accuracy of each model
#The 'mutate()' function is used to reorder the models in descending order of accuracy
#The 'coord_cartesian()' function is used to define a custom range for the y-axis
acc_plot <- results %>%
  mutate(Model = reorder(Model, desc(Accuracy))) %>%
  ggplot(aes(Model, Accuracy)) +
  geom_bar(stat = "identity", color = "black", fill = "#FF000050", width = 0.5) +
  xlab("Model Type") + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_cartesian(ylim = c(0.85, 1.0))

#Creating a bar plot visualizing the sensitivity of each model
#The 'mutate()' function is used to reorder the models in descending order of sensitivity
#The 'coord_cartesian()' function is used to define a custom range for the y-axis
sen_plot <- results %>%
  mutate(Model = reorder(Model, desc(Sensitivity))) %>%
  ggplot(aes(Model, Sensitivity)) +
  geom_bar(stat = "identity", color = "black", fill = "#FF000050", width = 0.5) +
  xlab("") + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_cartesian(ylim = c(0.85, 1.0))

#Creating a bar plot visualizing the specificity of each model
#The 'mutate()' function is used to reorder the models in descending order of specificity
#The 'coord_cartesian()' function is used to define a custom range for the y-axis
spec_plot <- results %>%
  mutate(Model = reorder(Model, desc(Specificity))) %>%
  ggplot(aes(Model, Specificity)) +
  geom_bar(stat = "identity", color = "black", fill = "#FF000050", width = 0.5) +
  xlab("") + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_cartesian(ylim = c(0.85, 1.0))

#Displaying all three of the above defined plots together in a rectangular fashion
grid.arrange(sen_plot, acc_plot, spec_plot, ncol = 3)
#CHUNK END

##CHUNK NAME: results_table

#CHUNK START
#Creating a table containing the values of the performance metrics of each model
#The '.[,c(5, 1, 2, 3, 4)]' statement is used to specify the order in which the columns of the table are to be displayed.
#Each number corresponds to the column index. Therefore translating to '.[,c(Rank, Model, Accuracy, Sensitivity, Specificity)]'
#The 'align = rep("c", 5)' argument is used to create a vector that specifies the alignment of each colum. In this case, as all columns are to be center aligned, the vector contains 5 occurences of "c", short of "center".
results %>%
  arrange(desc(Accuracy)) %>%
  mutate(rank = c(1, 1, 2, 3)) %>%
  .[,c(5, 1, 2, 3, 4)] %>%
  kable(digits = 4,
        align = rep("c", 5),
        col.names = c("Rank", "Model Name", "Accuracy", "Sensitivity", "Specificity"))
#CHUNK END

###SUBSECTION END

####SECTION END