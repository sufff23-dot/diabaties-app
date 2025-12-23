{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4717187d-a910-44ca-b2c9-677e86096ddd",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'logistic_regression_model.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 8\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Load the trained logistic regression model\u001b[39;00m\n\u001b[0;32m      7\u001b[0m model_filename \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mlogistic_regression_model.pkl\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[1;32m----> 8\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mopen\u001b[39m(model_filename, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrb\u001b[39m\u001b[38;5;124m'\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m file:\n\u001b[0;32m      9\u001b[0m     model \u001b[38;5;241m=\u001b[39m pickle\u001b[38;5;241m.\u001b[39mload(file)\n\u001b[0;32m     11\u001b[0m \u001b[38;5;66;03m# Define the title of the Streamlit app\u001b[39;00m\n",
      "File \u001b[1;32mD:\\Anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:324\u001b[0m, in \u001b[0;36m_modified_open\u001b[1;34m(file, *args, **kwargs)\u001b[0m\n\u001b[0;32m    317\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m file \u001b[38;5;129;01min\u001b[39;00m {\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m2\u001b[39m}:\n\u001b[0;32m    318\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    319\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIPython won\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mt let you open fd=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfile\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m by default \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    320\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mas it is likely to crash IPython. If you know what you are doing, \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    321\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124myou can use builtins\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m open.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    322\u001b[0m     )\n\u001b[1;32m--> 324\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m io_open(file, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'logistic_regression_model.pkl'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained logistic regression model\n",
    "model_filename = 'logistic_regression_model.pkl'\n",
    "with open(model_filename, 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Define the title of the Streamlit app\n",
    "st.title('Diabetes Prediction App')\n",
    "st.write('Enter the patient details below to predict the likelihood of diabetes.')\n",
    "\n",
    "# Define feature ranges and default values based on the dataset's descriptive statistics\n",
    "# These ranges are approximations and can be fine-tuned based on domain knowledge\n",
    "feature_info = {\n",
    "    'Pregnancies': {'min': 0, 'max': 17, 'default': 3, 'step': 1},\n",
    "    'Glucose': {'min': 40, 'max': 200, 'default': 120, 'step': 1},\n",
    "    'BloodPressure': {'min': 20, 'max': 122, 'default': 70, 'step': 1},\n",
    "    'SkinThickness': {'min': 7, 'max': 99, 'default': 29, 'step': 1},\n",
    "    'Insulin': {'min': 14, 'max': 846, 'default': 155, 'step': 1},\n",
    "    'BMI': {'min': 15, 'max': 67, 'default': 32, 'step': 0.1},\n",
    "    'DiabetesPedigreeFunction': {'min': 0.07, 'max': 2.42, 'default': 0.47, 'step': 0.001},\n",
    "    'Age': {'min': 21, 'max': 81, 'default': 33, 'step': 1}\n",
    "}\n",
    "\n",
    "# Create input widgets in the sidebar\n",
    "st.sidebar.header('Patient Input Features')\n",
    "\n",
    "input_data = {}\n",
    "for feature, info in feature_info.items():\n",
    "    if feature == 'BMI' or feature == 'DiabetesPedigreeFunction': # Use number_input for float values\n",
    "        input_data[feature] = st.sidebar.number_input(\n",
    "            f'{feature}',\n",
    "            min_value=float(info['min']),\n",
    "            max_value=float(info['max']),\n",
    "            value=float(info['default']),\n",
    "            step=float(info['step']),\n",
    "            format=\"%.3f\"\n",
    "        )\n",
    "    else: # Use slider for integer values, or number_input if step is 1 and range is large\n",
    "         input_data[feature] = st.sidebar.slider(\n",
    "            f'{feature}',\n",
    "            min_value=info['min'],\n",
    "            max_value=info['max'],\n",
    "            value=info['default'],\n",
    "            step=info['step']\n",
    "        )\n",
    "\n",
    "\n",
    "# Create a DataFrame from the input data, ensuring column order matches training data\n",
    "# The order of columns in X from the kernel state is: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age\n",
    "input_df = pd.DataFrame([input_data])\n",
    "input_df = input_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]\n",
    "\n",
    "st.subheader('User Input Features')\n",
    "st.write(input_df)\n",
    "\n",
    "# Make prediction\n",
    "prediction = model.predict(input_df)\n",
    "prediction_proba = model.predict_proba(input_df)\n",
    "\n",
    "st.subheader('Prediction Result')\n",
    "if prediction[0] == 0:\n",
    "    st.success(f'The model predicts: No Diabetes (Probability: {prediction_proba[0][0]:.2f})')\n",
    "else:\n",
    "    st.error(f'The model predicts: Diabetes (Probability: {prediction_proba[0][1]:.2f})')\n",
    "\n",
    "st.write('**Note:** A prediction of 0 means the model predicts \\'No Diabetes\\', and 1 means the model predicts \\'Diabetes\\'.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5915d769-7c20-4166-8832-ac67eeaea01e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
