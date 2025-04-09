from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from prophet import Prophet




#--metrics de useful_functions.py--

def calculate_metrics(y_true, y_pred):
    nrmse = mean_squared_error(y_true, y_pred, squared=False) / (y_true.max() - y_true.min())
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)  #
    r2 = r2_score(y_true, y_pred)
    return nrmse, mape, mae, r2

#--fonction inspiré du code de Sarimax.ipynb--



def find_best_sarimax(series_train, series_test, p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
    """
    Recherche le meilleur modèle SARIMAX en testant toutes les combinaisons possibles de paramètres.
    
    Args:
        series_train (DataFrame): Série temporelle d'entraînement avec une colonne 'sales'.
        series_test (DataFrame): Série temporelle de test avec une colonne 'sales'.
        p_values, d_values, q_values (range): Ordres non saisonniers.
        P_values, D_values, Q_values (range): Ordres saisonniers.
        s_values (list): Liste des périodes saisonnières.
    
    Returns:
        list: Liste des RMSE pour chaque combinaison.
        list: Liste des paramètres correspondants (ordre et ordre saisonnier).
    """
    rmse_list = []
    param_list = []
    
    # Générer toutes les combinaisons de paramètres
    param_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values))
    
    # Itérer sur chaque combinaison
    for params in tqdm(param_combinations, desc="Testing SARIMAX models"):
        try:
            order = (params[0], params[1], params[2])
            seasonal_order = (params[3], params[4], params[5], params[6])
            
            # Créer et ajuster le modèle
            model = SARIMAX(series_train['sales'], order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=False)
            
            # Prédictions sur le test set
            forecast = results.get_prediction(start=series_test.index[0], end=series_test.index[-1])
            y_pred = forecast.predicted_mean
            y_true = series_test['sales']
            
            # Calcul du RMSE
            rmse = mean_squared_error(y_true, y_pred)
            
            # Sauvegarde des résultats
            rmse_list.append(rmse)
            param_list.append({"order": order, "seasonal_order": seasonal_order})
        
        except Exception as e:
            # En cas d'erreur, afficher les paramètres concernés
            print(f"Erreur avec les paramètres {params}: {e}")
            continue
    
    return rmse_list, param_list




def find_best_auto_arima(series_train,m=7,exog_vars=None, seasonal=True):
    """
    Trouve le meilleur modèle SARIMA automatiquement en utilisant auto_arima.

    Args:
        series_train (DataFrame): Série temporelle d'entraînement avec les colonnes 'sales'.
        exog_vars (list or None): Liste des noms des colonnes exogènes. Par défaut, None.
        m (int): Période saisonnière (par défaut 7 pour hebdomadaire).
        seasonal (bool): Indique si le modèle doit être saisonnier (par défaut True).

    Returns:
        dict: Résumé du meilleur modèle et ses paramètres.
    """
    try:
        exog_data = series_train[exog_vars] if exog_vars else None


        auto_model = auto_arima(
            series_train['sales'],
            exogenous=exog_data,
            seasonal=seasonal,
            m=m,
            trace=True,
            suppress_warnings=True
        )

        model_summary = auto_model.summary()
        
        best_params = {
            "order": auto_model.order,
            "seasonal_order": auto_model.seasonal_order,
            "aic": auto_model.aic()
        }

        return {"summary": model_summary, "best_params": best_params}

    except Exception as e:
        print(f"Erreur lors de l'exécution de auto_arima : {e}")
        return None


def fit_and_forecast_sarimax(cluster_train, cluster_test, p, d, q, P, D, Q, s):
    """
    Ajuste un modèle SARIMAX sur les données d'entraînement et retourne les prévisions pour l'entraînement et le test.

    Args:
        cluster_train (DataFrame): Ensemble d'entraînement avec une colonne 'sales' et un index temporel.
        cluster_test (DataFrame): Ensemble de test avec une colonne 'sales' et un index temporel.
        p, d, q (int): Paramètres non saisonniers du modèle SARIMAX.
        P, D, Q (int): Paramètres saisonniers du modèle SARIMAX.
        s (int): Période saisonnière.

    Returns:
        tuple: (train_forecast_values, test_forecast_values)
    """
    # Extraire les ventes de l'ensemble d'entraînement
    y_train = cluster_train['sales']

    # Ajustement du modèle SARIMAX
    model_sarimax = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model_sarimax.fit(disp=False)

    # Prévisions pour l'ensemble d'entraînement
    train_forecast = results.get_prediction(
        start=cluster_train.index[0],
        end=cluster_train.index[-1]
    )
    train_forecast_values = train_forecast.predicted_mean.values

    # Prévisions pour l'ensemble de test
    forecast_steps = len(cluster_test)
    test_forecast = results.get_forecast(steps=forecast_steps)
    test_forecast_values = test_forecast.predicted_mean.values

    return train_forecast_values, test_forecast_values




def data_stream(test_data, chunk_size):
    for i in range(0, len(test_data), chunk_size):
        yield test_data.iloc[i:i + chunk_size]


def rolling_prophet_forecast(series_train, series_test, vars, chunk_size, window_size):
    """
    Effectue des prévisions avec une fenêtre glissante à l'aide de Prophet.
    
    Args:
        series_train (DataFrame): Données d'entraînement avec les colonnes spécifiées dans `vars`.
        series_test (DataFrame): Données de test avec les colonnes spécifiées dans `vars`.
        vars (list): Liste des colonnes à utiliser, ex. ['date', 'sales'].
        chunk_size (int): Taille des prévisions futures (par exemple, 7 pour hebdomadaire).
        window_size (int): Taille de la fenêtre glissante utilisée pour l'entraînement (par exemple, 365 pour une année).
    
    Returns:
        list: Liste des prévisions pour chaque fenêtre glissante.
    """
    new_train_data = series_train[vars].reset_index(drop=True)
    new_train_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

    predictions = []

    for new_data in data_stream(series_test, chunk_size):

        new_data = new_data[vars].reset_index(drop=True)
        new_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

        new_train_data = pd.concat([new_train_data, new_data], ignore_index=True).tail(window_size)

        model_prophet = Prophet()
        model_prophet.add_seasonality(name='weekly', period=7, fourier_order=3)
        model_prophet.add_seasonality(name='annual', period=365, fourier_order=10)

        model_prophet.fit(new_train_data)


        future = model_prophet.make_future_dataframe(periods=chunk_size)
        forecast = model_prophet.predict(future)

        y_test_pred = forecast.tail(chunk_size)['yhat'].values
        predictions.append(y_test_pred)

    return predictions
