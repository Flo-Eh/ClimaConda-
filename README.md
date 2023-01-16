## Presentation

This repository contains the code for our project **ClimaConda**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).


Based on a research paper published in 2022, this project aims to link environmental issues with Machine Learning by measuring and predicting greenhouse gas emissions for the next 10 years. The paper in question tries to find the best possible model to evaluate these emissions in India. Its academic advantage therefore lies in the possibilitý to test different time series models: statistical (ARIMA, SARIMAX, HOLT-WINTER), Machine Learning (Linear Regression, Random Forest), and Deep Learning (LSTM). Moreover, the approach allows to have a deep approach on the different indicators of error measures (MSE, RMSE, MLSE etc...) in order to compare the predictions according to the models.


This project was developed by the following team :

- Mai-Anh Vu ([GitHub](https://github.com/mai-anh-vu) / [LinkedIn](https://www.linkedin.com/in/mai-anh-vu-3449505/))
- Michael Laidet ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Florian EHRE ([GitHub](https://github.com/Flo-Eh) / [LinkedIn](https://www.linkedin.com/in/florian-ehre-01323395/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

You can access the app by following this link:
https://flo-eh-climaconda--streamlit-appclimacondav3-jans2h.streamlit.app/



To run the app :

```shell
cd streamlit_app
conda create --name appclimacondav3 python=3.9
conda activate appclimacondav3
pip install -r requirements.txt
streamlit run climacondav5.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
