# Project5

**IDEAL PROJECT**


**Predicting risk of Stroke in patients using Data from EHRs**

**Motivation**

Stroke is the leading cause of disability in US.80% of strokes can be prevented and 58% of Americans don’t know if they are at risk.

The main aim of my project is to to use NLP to analyze the medical records and predict the risk of stroke in patients.I plan to use the MIMIC-III data (Medical Information Mart for Intensive Care), a large, single-center database comprising information relating to patients admitted to critical care units at a large tertiary care hospital.

**Part 1**

The  MIMIC-III database includes 26 tables.I plan to focus on 3 tables - the NOTEEVENTS (including nursing and physician notes, ECG reports, radiology reports, and discharge summaries), DIAGNOSES_ICD (Hospital assigned diagnoses, coded using the International Statistical Classification of Diseases and Related Health Problems (ICD) system) and D_DIAGNOSES_ICD (Dictionary of International Statistical Classification of Diseases and Related Health Problems (ICD-9) codes relating to diagnoses) tables. After data cleaning and processing, I will extract the records that are labeled as stroke and non-stroke based on the ICD9 code and the patients' past medical, social and history information. Then I will combine medical records for stroke patients and ( randomly sampled) records for non-stroke patients for cosine similarity and component analysis.I then plan to apply the principle component analysis  to visulize the how stroke and non-stroke samples are separated.

**Part 2e**

I then plan to apply different prediction models (LogisticRegression ,Support Vector Machine classifier etc )  to predict the diagnosis of stroke. I plan to follow the following steps: 

1) split the data into train data and test data
2) use NLP  to transform text notes, 
3) perform lasso feature selection using linear support vector classifier, 
4) optimize the LogisticRegression and Support Vector Machine classifiers,
5) compare the performance of the classifiers 

**Part 3**

I then plan to build an app to predict the risk of stroke. Health insurance company may use this tool to perform prescreen, save lives and decreased costs.



**BACK UP PROJECT**

There is a chance that I might not get access to the data for my ideal project in time (I will still be completing my ideal project).So I have a back up project in mind for project 5.

For my back up project, I plan to use the  "Allstate Purchase Prediction Dataset" on Kaggle. 

**Motivation**

When someone has been through  a serious car accident, their main focus is on the things that matter the most: family, friends, and other loved ones. Filing insurance paperwork should be the the last thing on their mind.So the main aim of my project is 2 fold: first to predict the severity of insurance claims, and second to help the company to evaluate potential clients based on the importance of predictors.This analysis would potentially help the company to improve their customer care and claims management service.

**Dateset**

The dataset contains the history of car insurance quotes that each customer reviewed before making a purchase, the options they actually purchased (the "purchase point"), and data about the customer and their car. The datset contains a training data set with 188,318 rows and a testing data set with 125,546 rows. The training data set consists of 188,318 observations and 131 variables which include 72 binary categorical variables, 43 non-binary variables with 3-326 levels, 14 continuous variables and the outcome variable, "loss".

**Part 1**

After data cleaning and processing, I plan to fit several models and then select the best model and optimize it.I plan to do all the analysis in PySpark to get learn more about the methodology.

**Part 2**

After fitting the best optmized model on the testing set, I plan to build an app that would predict the severity of a claim.It would be helpful to the company to make better business decisions.



