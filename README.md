# TreeSHAP for KNIME

## Introduction
This repository contains the predictor nodes for the TreeSHAP KNIME extension. SHAP (SHapley Additive exPlanations) is a game 
        theoretic approach to explain the output of any machine 
        learning model. It connects optimal credit allocation with 
        local explanations using the classic Shapley values from 
        game theory and their related extensions. While SHAP can 
        explain the output of any machine learning model, 
        Lundberg and his collaborators have developed a high-speed 
        exact algorithm for tree ensemble methods <a href="https://github.com/slundberg/shap">[1]</a>, 
        <a href="https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html">[2]</a>.
		
## Usage 

The Tree SHAP Random Forest Predictor is used as a substitute to the
		Random Forest Predictor. Simply replace every Random Forest Predictor with this 
		node to get started. If you are using a different tree based method, consider the 
		other nodes in this package.

## Interpretation
The beautiful thing about SHAP values is the intuitive interpretation.
		Every model has an expected output, the average prediction. The model prediction 
		for a data row is the expected output plus the summation of SHAP values.
		This leads to intuitive explanations, for example in predictive maintenance 
		"The high production output over the last three months contributed +20% probability
		that the machine breaks down in the next month.".

## Enterprise Support
If you need help integrating explainable machine learning methods
		 in your company, please contact me at morriskurz@gmail.com