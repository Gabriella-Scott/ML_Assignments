# Machine Learning 441/741 — Assignment 2

## Nearest Neighbours & Decision Trees

**SU Corporate Vertical**  
**Total:** [170]  
**Deadline:** 4 September 2026, 08:00

## Instructions

When completing this assignment, follow the instructions given below:
- The assignment must be completed by each student individually.
- You may use any programming language to complete the assignment, and you may make use of any
libraries, including machine learning libraries.
- You have to write a report, using the IEEE conference template (google!), in two-column, 10pt format.
Please see section 5 for tips on report writing. Also see the writing rules – uploaded to your STEMlearn module – for a list of writing rules. Please consult these writing rules and apply them. Note that only the report will be evaluated, not your code. Therefore, a bad report will result in bad marks.
- Submit your report in pdf format. Note that no format other than pdf will be accepted. Make sure that
your report has a reference to your git repository for your code. Code will not be evaluated, but may be scrutinized if found necessary. Make sure that you name your report file as ????????RWxxxassignment2.pdf, where you replace the question marks with your student number, and xxx with the module code that you are registered for. Please note that I use a script to pull out all reports, and if you do not follow this file naming convention, your report will not be extracted for evaluation.
- Make sure that your name, surname and student number are clearly indicated in the front matter (title
section) of your report. If there is no identification of the author of the report in the front matter of your report, your report will not be evaluated.
- Upload your report via STEMlearn before the deadline of 6 September 2026, 08:00. Note that late
assignments will not be accepted. After this deadline, I will extract all reports to start with evaluation of the reports that day.

## 1. Purpose of the Assignment

The main objectives of this assignment are to test your ability to develop classification models for the provided dataset, to identify data quality issues that need to be addressed for the classification models employed, to identify and implement only the necessary data transformations for the employed classification models, to compare the performance of the implemented models, and to write a report. This assignment focuses on only two machine learning classification models, namely k-nearest neighbours and classification trees. With reference to the machine learning models, you may use any libraries to implement these models. You may also use any libraries to perform the necessary pre-processing and statistical analysis of the performance of the models. If you do make use of libraries, please provide details of these libraries in the appropriate section of your report. With reference to the report, you have to write a report wherein you provide responses in clear narrative on the aspects enumerated below, under appropriate section headings. Note that figures and tables provided in the report must be referred to and discussed, otherwise they will not be looked at nor marked.

## 2. The Dataset: UNSW NB15 Network Traffic

You have been provided with a dataset for assignment 1, with the goal to identify data quality issues. This dataset was an edited version of the UNSW-NB15 network traffic dataset, with various additional data quality issues explicitly injected within that dataset. This section provides detail on the original network traffic dataset, the data quality issues that were present in the assignment 1 version of the dataset, and then a short explanation of data quality issues that remain in the dataset provided for assignment 2. Details on the UNSW NB15 Network Traffic Dataset The dataset contains information about network packet data in aid of creating models to predict and identify cyberattacks. The dataset contains a mix of real modern normal internet activity, as well as synthetic contemporary attack activity. Each instance of data describes a single packet, which includes information on IP addresses, transaction protocol, size, time, and more. The problem is to classify each instance into one of ten attack types, namely normal, Fuzzers, Reconnaissance, Shellcode, Analysis, Backdoors, DoS, Exploits, Generic, or Worms. For more detail on the original dataset and its descriptive features, please refer to the uploaded file UNSW-NB15 features.csv. The Data Quality Issues The dataset that you have received for assignment 1 was a doctored version of the network traffic dataset, with a number of data quality issues injected into this dataset. A list of these injected data quality issues is provided below, in no particular order:
- A1 has a few incorrect negative values; most of the values for this feature are positive, so the negative
values raises a concern.
- A7 is irrelevant – all values are randomly selected from a uniform distribution.
- A10 has invalid character values for this numerical descriptive feature.
- A11, A22, A24, A25, A28, A29, A32, A42, A45 have missing values.
- The missing values are weakly correlated to one another.
- A13 values are order of magnitude larger than that of the other descriptive features.
- A17 and A18 have cardinality of 1
- A46 has potential irregular cardinality due to an invalid categorical value; most of the values are binary,
with very few incorrect values set to 2 instead of binary.
- The target feature has some missing values.
- The target feature has a skew class distribution.
Though out of scope for assignment 1, it might have been noted that some instances have too many missing features. Additionally, if the features A3, A4, and A5 have been one-hot encoded, there were several very strong correlations which could have been discovered between particular values of each feature, and other original features. The dataset also contains several data quality issues that are present in the original, undoctored version of the dataset. An inexhaustive list of these original data quality issues is provided below:
- service has many missing values.
- sbytes and sloss are perfectly correlated.
- dbytes and dloss are perfectly correlated.
- id is a unique descriptive feature.
- is ftp login and ct ftp cmd are perfectly correlated.
- There are many other very strong correlations.
- Many features have large positive outliers.
Revised Network Traffic Dataset The dataset uploaded to STEMlearn as networkTraffic.csv is an altered version of the dataset uploaded for assignment 1, with some of the added data quality issues removed. Note that all of the original data quality issues present in the dataset remain. The following statements are true of the networkTraffic.csv:
- There remain features with missing values, with missing values indicated by the character symbol ‘?’.
- There remain many features with correlations.
- There remain features with outliers.
- There remain features with numeric ranges that differ significantly from one another.
- There are numerical and categorical features.
- The features proto, state, and service are nominal.
- Feature id has a unique value for each observation.
- The class distribution remains skew.

## 3. Tasks to Complete

Complete the assignment in the following steps:
1. Download the networkTraffic.csv dataset, which is a comma delimited dataset. You have to use this
dataset and not the dataset available online. Note that the target feature is the last feature, and that there are a total of 43 descriptive features.
2. Provide in your report a discussion of k-nearest neighbours and classification trees.
3. Follow this discussion with your expectations of the impact of the different remaining data quality issues
for each of the two machine learning models.
4. For each of the machine learning approaches, discuss the data-preprocessing steps that you have implemented to optimally transform the dataset for that specific machine learning approach and to correct
data quality issues. Note: do not do unnecessary data transformations. Carefully think about the data transformations needed for each of the machine learning algorithms, and apply only those. Provide justifications for each of these pre-processing steps. Should you decide not to address a data quality issue, justify this decision. Note that the bulk of the marks are for your justifications.
5. Develop the k-nearest neighbours and classification trees models and evaluate the performance of the two
models on your pre-processed dataset. Make sure to construct optimal configurations of your models both with respect to architecture and values for control parameters. Describe the process that you have followed to produce an optimal configuration for each model. For this purpose, carefully decide on the performance metrics that you will use. Conclude on which one of the two approaches is best for this problem, and support your conclusion with justifications. For the purposes of this assignment, make sure to report the performance based on a k-fold cross-validation. Decide on the number of folds with a justification.

## 4. Mark Rubric

The assignment will be assessed as follows: Aspect Mark Title & Abstract 4 Introduction 8 Background Machine learning algorithm description 8 Expectations wrt data quality issues 18 Implementation 8 Empirical process Data pre-processing 18 Control parameter tuning 8 Performance metric 4 Analysis process 8 Results & discussion 40 Conclusions 4 References 4 Linguistic quality 38 Total 170

## 5. Report Writing

The following is a general guideline of how to structure your report. Title Section Provide your report with a title, and as author provide your initials, surname and student number. Also provide an email address. Abstract Provide a very concise summary of what this report provides. Provide some context, the goals, how these were achieved, and the main observation. The abstract should be short. No more than 300 words. The purpose of the abstract is to convince the reader to continue reading your report.
### 6. Introduction
The introduction sets the stage for the remainder of your report. You usually have very general statements here. The introduction prepares the reader for what to expect from reading your report. In general, the introduction should be a summary of your entire report. Start by stating the context, moving towards the goals. Then elaborate on how these goals have been obtained, what you have done. Give a motivation for why this is done. Summarize the main observations of the study. You basically give a teaser to the reader, to convince the reader to continue reading the report. Give an outline of the remainder of the report.
### 7. Background
A very high level discussion on the problem domain and the algorithms and/or approaches that you have used. Do not be too specific on the algorithms and approaches. This section is typically where the “base cases” of concepts that appear throughout the remainder of your report are discussed. It is also an ideal place to refer a reader to other sources containing relevant information on the topic but which is outside the scope of your assignment. It is the perfect place for pseudo code of existing approaches. Remember to discuss very generally. After reading this section the marker should be able to determine whether or not you know what you’re talking about. Keep in mind that this is a background section, and does not contain any detail on what you have done, but only provides a summary of related background to understand what you have done.
### 8. Methodology
In this section you discuss how you have approached, implemented and solved your assignment problem. You provide pseudo code where necessary (only for new algorithms) and discussions of the solutions that you have implemented. This is also the section where your discussion specializes on the concepts mentioned in the background section. Be very specific in your discussions in this section, to clearly describe what you have done and how you have done it.
### 9. Empirical Procedure
Here you describe the empirical procedure followed to apply your algorithms to obtain answers to the goals/hypothesis of the study. You elaborate on the performance measures used and provide the benchmark problems used. Provide all control parameter values with a motivation for why you have used these, and state the number of independent runs. If statistical tests are used, these are discussed here. After reading this section (in addition to the background) the reader should be able to duplicate your experiments to obtain similar results to those obtained by you.
### 10. Research Results
This is the section where you report your results obtained from running the experiments as discussed in the implementation section, using the empirical procedure above. You have to give, at least, averages and standard deviations for the experiments/simulations. Thoroughly discuss the results that you have obtained and provide clear arguments in support of your results and observations from these results. Answer questions like “are these results to be expected?”, “why did these results occur?” and “would different circumstances lead to different results?”.
### 11. Conclusion
Start this section by stating again the goals of the report, what was done and how. Very general conclusions about the assignment that you have done are given. This section “answers” the questions and issues that you have raised and investigated. This is the final section in your document so be sure that all the issues raised up until now are answered here. This is also the perfect section to discuss what you have learnt in doing this assignment, and to provide any ideas for future work. References Provide all references that you have consulted.
