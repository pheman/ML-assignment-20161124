
Repository
===========
  Current repository created to provide data and description for MSXF ML test assignment.

The repository includes the following files:
  
  * "Readme.md" - current file, describing repository and assignmment task
  * "samples_codebook.part1..3.rar" contains 3 files:
    * "development_sample.xlsb" - development sample. use to build model.
    * "assessment_sample.xlsb" - final assessment sample.  
    * "Codebook.xlsb" -  partial description for "development_sample.xlsb" and "assessment_sample.xlsb"

Introduction
===========
  Current assignment task will require you to build model to predict given binary target variable. 

You will download 2 data samples. 1st one contains all data necessary to develop your model including target variable(field "target"),
2nd data sample is for final assessment purpose, it will contain same fields except for target variable.
In the end you will need to deliver both samples with all transformations made plus 1 prediction field
which have to contain predicted probability of target value = "1".

Additionally general description required as well as code\scripts used in the process(codes are optional).



Tips:
  1. if for some reason you cant finish task you still can submit unfinished materials\descriptions. Current task is important
but not the only assessment criteria.  
  2. there is no restrictions on tools you can use for preprocessing or modelling


Data
===========
  Data you will use is simplified version of what one may expect to have as material to work with in consumer finance,
specifically that is an imitation of around 500 000 consumer loan deals signed during period from SEP 2004 to FEB 2005.
Target variable is imitation of BAD - GOOD loan repayment performance, where target = "1" defines BAD performed clients.
Some preprocessing already made to the variables.
Variables description can be partially found in Codebook.
  
Assignment task:
===========
  In order to complete this assignment, you must do the following:
  1. Prepare data and build the model using "development_sample.xlsb" to train and test your model, 
considering field "target" as target(dependent) variable and the rest as predictors(independent).
  2. Predict the probability of target variable to have value="1" for both "development_sample.xlsb" and "assessment_sample.xlsb"
  3. Create 2 files(development+assessment) containing fields: predicted probability + "ID2" from original file + all fields used in final model(with transformations if any made)
  4. Create 1 description file with basic description on preprocessing, variable selection, modelling etc.
  5. (optional) create file with code/scripts used.
  6. Send or save files 3 + 1(optional) files for further assessment
