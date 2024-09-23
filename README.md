**Property Price Prediction Project**

**Project Overview**
This project is an end-to-end development aimed at predicting property prices in the Federal Capital of Argentina. It encompasses the entire process from data extraction to transformation, model training and validation, deployment, and ongoing monitoring. The target audience includes both end-users interested in determining whether a property price aligns with market values and real estate agencies seeking a quick quotation for properties.

**Project Structure**
Each stage of the project is organized into separate folders:

- **ETL:** This stage handles the extraction of property price data by making requests to the Mercado Libre API. Approximately 80,000 samples are extracted, processed, and stored in a MySQL database.

- **Training:** This folder contains a Jupyter notebook that imports the data obtained in the previous step. It includes exploratory data analysis (EDA) to understand the data, uncover significant insights, and examine variable relationships. Data processing steps such as outlier detection and treatment, null value imputation, categorical variable handling, feature engineering, and data scaling are implemented. These steps are encapsulated in a pipeline, which will be utilized during deployment. The models are then trained using cross-validation to optimize hyperparameters derived from a grid search. A Gradient Boosting Machine (GBM) model demonstrates a favorable bias-variance tradeoff with a mean absolute error (MAE) of approximately 40,000 and a mean absolute percentage error (MAPE) of 18%.

- **Model:** This folder contains the trained model, the pipeline for transforming data as required by the model, and the residual values (MAE and MAPE) of the model.

- **Deployment-DEV:** This section includes the development of a Flask API that serves as the backend to ensure the model operates correctly.

- **Deployment-PROD:** This folder contains the production backend developed with FastAPI, housed within a Docker container.

- **Client:** This part features the user interface created with Streamlit, allowing users to request property quotations. It sends requests to the backend and receives predictions for property prices, also running within a Docker container.

- **Model Monitoring:** This notebook facilitates model monitoring. It was evaluated two weeks after training, yielding an MAE of 26,000 and a MAPE of 16%—better values than those achieved during training.

- **Utils:** This folder contains the necessary classes and functions to execute the aforementioned components.

**Conclusion**
The project is deployed on a server using Google Cloud Platform (GCP) with a Linux operating system, making it accessible for anyone interested in predicting property prices.

**Screenshots**
*DEV ENVIRONMENT*: by Postman 
![dev_example](https://github.com/user-attachments/assets/2c98c4d8-8e96-4839-8a9b-2e9aecc848f3)

*PROD ENVIRONMENT*: by a custom UI
PREDICTION:
![prod_example_1](https://github.com/user-attachments/assets/3bc2d563-a7fb-4b46-9cec-032fddce2357)
![prod_example_2](https://github.com/user-attachments/assets/a7522369-8a41-414d-a7c3-ad7d5b50064d)

REAL:
![prod_example_3](https://github.com/user-attachments/assets/2ceddbdc-ed7b-445f-baa6-436a849ba4c3)
