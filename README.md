# Prédiction des Ventes - Séries Temporelles Hiérarchiques

## Description du Projet

Ce projet vise à prédire les ventes en utilisant des modèles de prédiction de séries temporelles globales et des méthodes de prédiction hiérarchique. Il repose sur l'application de différents modèles de machine learning et deep learning pour fournir des prévisions fiables et optimisées, tout en assurant la cohérence des prédictions à travers les niveaux de la hiérarchie.

## Approche

### 1. Prédiction Globale

Plusieurs modèles sont testés pour capturer la dynamique des ventes :

- **SARIMAX** (Saisonality AutoRegressive Integrated Moving Average with Exogenous variables)
- **Prophet** (modèle bayésien de Facebook)
- **XGBoost** (gradient boosting)
- **Random Forest** (arbres de décision)
- **LSTM** (Long Short-Term Memory, réseaux de neurones récurrents)

### 2. Prédiction Hiérarchique

Les prévisions sont effectuées à plusieurs niveaux de granularité et réconciliées pour assurer la cohérence des résultats :

- **Approche Bottom-Up** : agrégation des prévisions faites à un niveau inférieur pour obtenir celles des niveaux supérieurs.
- **Clustering de Séries Temporelles** : segmentation des séries temporelles pour améliorer les performances des modèles.
- **Réconciliation des Prévisions** : techniques de correction pour aligner les prévisions avec la structure hiérarchique des données.

## Données

Les données utilisées pour ce projet proviennent du défi Kaggle **Store Sales - Time Series Forecasting**. Elles sont accessibles via le lien suivant :
[Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting/data)

## Utilisation

1. **Chargement des données**
2. **Exploration et preprocessing**
3. **Application des modèles globaux**
4. **Clustering et approche hiérarchique**
5. **Réconciliation des prévisions**
6. **Analyse des performances**

##
