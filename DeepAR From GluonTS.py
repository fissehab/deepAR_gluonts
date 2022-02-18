# Databricks notebook source
!pip install --upgrade mxnet==1.6.0
!pip install gluonts


# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
!unzip LD2011_2014.txt.zip

# COMMAND ----------

!head LD2011_2014.txt

# COMMAND ----------

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
import numpy as np

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

# COMMAND ----------

df = pd.read_csv('LD2011_2014.txt', sep = ';', index_col = 0, parse_dates = True, decimal = ',')

# COMMAND ----------

plt.figure(figsize=(12, 6), dpi=100, facecolor="w")
for col in df.columns[:10]:
    plt.plot(df[col], label = col)

plt.ylabel("Energy Consumptions")
plt.xlabel("Date")
plt.title("Energy Consumptions Per House Hold")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4)
plt.show()

# COMMAND ----------

df.head()

# COMMAND ----------

fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=True)
axx = axs.ravel()
for i in range(0, 10):
    df[df.columns[i]].loc["2014-12-01":"2014-12-14"].plot(ax=axx[i])
    axx[i].set_xlabel("date")    
    axx[i].set_ylabel("kW consumption")   
    axx[i].grid(which='minor', axis='x')

# COMMAND ----------

df_input = df.reset_index(drop=True).T.reset_index()

# COMMAND ----------

df_input.head()

# COMMAND ----------

ts_code=df_input["index"].astype('category').cat.codes.values
ts_code[0:7].reshape(-1,1)

# COMMAND ----------

df_train=df_input.iloc[:,1:134999].values
df_test=df_input.iloc[:,134999:].values

# COMMAND ----------

df_train.shape

# COMMAND ----------

df_test.shape

# COMMAND ----------

freq="15min"
start_train = pd.Timestamp("2011-01-01 00:15:00", freq=freq)
start_test = pd.Timestamp("2014-11-07 05:30:00", freq=freq)
prediction_lentgh=672

# COMMAND ----------

estimator = DeepAREstimator(freq=freq, 
                            context_length=672,
                            prediction_length=prediction_lentgh,
                            use_feat_static_cat=True,
                            cardinality=[1],
                            num_layers=2,
                            num_cells=32,
                            cell_type='lstm',
                            trainer=Trainer(epochs=5))

# COMMAND ----------

df_train[0:7].shape

# COMMAND ----------

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_train,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, fsc) in zip(df_train[0:7],
                             ts_code[0:7].reshape(-1,1))
], freq=freq)

test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_test,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, fsc) in zip(df_test,
                            ts_code.reshape(-1,1))
], freq=freq)

# COMMAND ----------

next(iter(train_ds))

# COMMAND ----------

predictor = estimator.train(training_data=train_ds)

# COMMAND ----------

from gluonts.evaluation.backtest import make_evaluation_predictions

# COMMAND ----------

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  
    predictor=predictor,  
    num_samples=100, 
)

# COMMAND ----------

from tqdm.autonotebook import tqdm

print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(df_test)))
print("Obtaining time series predictions ...")
forecasts = list(tqdm(forecast_it, total=len(df_test)))

# COMMAND ----------

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = prediction_lentgh
    prediction_intervals = (80.0, 95.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    plt.show()

# COMMAND ----------

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = prediction_lentgh
    prediction_intervals = (80.0, 95.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()

# COMMAND ----------

for i in tqdm(range(6)):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry)

# COMMAND ----------

from gluonts.evaluation import Evaluator
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(df_test))

# COMMAND ----------

item_metrics

# COMMAND ----------

agg_metrics
