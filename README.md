# COMP3000
Exploring Deep Learning models to diagnose Autism Spectrum Disorder (ASD)

## Assessing Feature Importance and Redundancy in Deep Learning based Autism Diagnosis

#### Links
	Source Code: https://github.com/v1dya/COMP3000
#### Project Vision
Convolutional neural networks (CNNs), a rapidly developing field in machine learning, show promise in defining the neural biomarkers for diagnosing Autism Spectrum Disorder (ASD). For these kinds of computational projects, the Autism Brain Imaging Data Exchange (ABIDE) dataset, which contains rich neuroimaging data, is a powerful resource. Nevertheless, the dataset's possible burden of redundant or less useful features could make the model difficult to interpret and reduce its capacity to provide the best possible diagnosis.

The aim of this project is to understand the importance of the different features in the ABIDE dataset while also evaluating which among them are redundant. By using these findings models can be made that are more efficient and more accurate than the ones that currently exist. It will help in increasing model interpretability and help realise what data we should focus on getting more. This project will use the Remove and Retrain (ROAR) approach to evaluate the different features in ABIDE.

The main objectives of this research project are:
1. Enhancing Model Efficiency
	-  Identifying and removing redundant features will make models cheaper to train and more efficient to run.
2. Improving explainability of a model's diagnosis
	- One of the main factors as to why CNNs are not used in clinical situations are their black box nature. This study aims to highlight the important features used by the model for it's diagnosis giving clinical professionals an idea of what the model used to make it's decision.
3. Contribute to the evolution of Deep Learning applications in medical diagnoses
	- Help future researchers in this topic use this study as an example on how to make deep learning models interpretable and help improve model interpretability methods.
#### Risk Plan

| Risk | Description | Mitigation |
|------|-------------|------------|
|Computational Resources|Training a CNN on a large dataset like ABIDE will need a considerable amount of computational resources that can be hard to get|Use university's resources to train the models or find cheap cloud computing services|
|Time Management|Other modules and work outside of university may eat into the time needed to complete this project due to unforeseen circumstances.|Use established time and project management approaches like Scrum and iterate over the project plan as and when needed|
|Limited Expertise|I might have trouble understanding advanced topics and methodologies due to having less experience| Take advice and resources from experienced peers and professors|
|Technical Challenges|Implementing and understanding ROAR might be harder than initially anticipated|Practice using the Python and read more literature on how others have used ROAR, if it's too difficult take a step back learn the topic and try again|
|Quality of Results|The results may not be as conclusive or clear as originally thought which could affect the paper's chances for a publication.|Repeat experiments and maintain high standard of methodology. Double check everything|

#### Proposed Gantt Chart
![image](https://github.com/v1dya/COMP3000/assets/55044178/11e1fdf0-3dcb-4957-9859-56dd5c3d2b25)
#### Tasks:
1. Literature Review:
	- Understand current methodologies and findings in ASD diagnosis using ML and the applications of ROAR by conducting a literature review
2. Data Collection and Preprocessing:
	- Obtain the ABIDE dataset and preprocess it to ensure it's ready for training the CNN.
3. Initial Model Training:
	- Get a baseline performance by training CNN with the complete ABIDE dataset.
4. Feature Importance Estimation:
	- Estimate the importance of features in the dataset with methods like SHapley Additive exPlanations (SHAP)
5. Implementing ROAR:
	- Iteratively remove features and retrain the model using ROAR method.
6. Performance Analysis:
	- Compare the performance of CNNs with the reduced features against the baseline CNN.
7. Identify Redundant Features and Critical Features:
	- Identify redundant features and features that cause a large drop in performance and document them
8. Draft of Report:
	- Draft a report with introduction, methodology, discussion, conclusion and further research.
9. Review and revision of report:
	- Have peers and advisors review the report and implement the feedback received from them.
10. Submission:
	- Submit report on time and submit to relevant academic journals.

