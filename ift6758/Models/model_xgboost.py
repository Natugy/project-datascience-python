"""
Partie 5 - Modèles avancés XGBoost avec suivi Wandb et génération de figures d'évaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import wandb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')


class XGBoostModelTrainer:
    """Classe pour entraîner et évaluer des modèles XGBoost avec suivi Wandb."""
    
    def __init__(
        self,
        project_name: str = "IFT6758-2025",
        experiment_name: str = "xgboost-baseline",
        wandb_entity: Optional[str] = "IFT6758-2025-A04",
        use_scaler: bool = True,
    ):
        """
        Initialise le trainer XGBoost.
        
        Args:
            project_name: Nom du projet Wandb
            experiment_name: Nom de l'expérience
            wandb_entity: Entité Wandb (team)
            use_scaler: Si True, applique StandardScaler aux features
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.wandb_entity = wandb_entity
        self.model = None
        self.scaler = StandardScaler() if use_scaler else None
        self.use_scaler = use_scaler
        self.run = None
        
    def init_wandb(self, config: Dict[str, Any]):
        """Initialise une run Wandb."""
        # Clé API Wandb
        os.environ["WANDB_API_KEY"] = "13a4f31490980ca10265e2cbf4d46f26ba7a9a7b"
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print("Warning: WANDB_API_KEY not found in environment variables")
            
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            entity=self.wandb_entity,
            config=config,
            reinit=True
        )
        
    def _scale_features(self, X_train, X_val, fit_scaler=True):
        """Applique la standardisation si activée."""
        if not self.use_scaler:
            return X_train, X_val
        
        if fit_scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        
        return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index), \
               pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str] = ["distance_net", "angle_net"],
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ) -> xgb.XGBClassifier:
        """
        Entraîne un XGBoost baseline avec distance et angle uniquement.
        
        Args:
            scale_pos_weight: Poids pour compenser le déséquilibre des classes
            
        Returns:
            Modèle XGBoost entraîné
        """
        config = {
            "model_type": "xgboost_baseline",
            "features": features,
            "n_features": len(features),
            "random_state": random_state,
            "use_scaler": self.use_scaler,
            "scale_pos_weight": scale_pos_weight,
            "hyperparameters": "default"
        }
        
        self.init_wandb(config)
        
        # Sélectionner et standardiser les features
        X_train_subset = X_train[features]
        X_val_subset = X_val[features]
        X_train_scaled, X_val_scaled = self._scale_features(X_train_subset, X_val_subset, fit_scaler=True)
        
        # Entraîner avec paramètres par défaut + scale_pos_weight
        params = {
            'random_state': random_state,
            'eval_metric': 'logloss'
        }
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
            
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
            verbose=False
        )
        
        # Log metrics
        self._log_metrics(X_train_scaled, y_train, X_val_scaled, y_val, "baseline")
        
        return self.model
    
    def train_with_all_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[Dict] = None,
        search_type: str = "grid",
        cv: int = 5,
        n_iter: int = 50,
        random_state: int = 42,
    ) -> xgb.XGBClassifier:
        """Entraîne XGBoost avec toutes les features et hyperparameter tuning."""
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
            }
        
        config = {
            "model_type": "xgboost_tuned",
            "n_features": X_train.shape[1],
            "search_type": search_type,
            "cv_folds": cv,
            "use_scaler": self.use_scaler,
        }
        
        self.init_wandb(config)
        
        # Standardiser les features
        X_train_scaled, X_val_scaled = self._scale_features(X_train, X_val, fit_scaler=True)
        
        base_model = xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')
        
        SearchClass = GridSearchCV if search_type == "grid" else RandomizedSearchCV
        search_params = {'cv': cv, 'scoring': 'roc_auc', 'n_jobs': -1, 'verbose': 0}
        if search_type == "random":
            search_params.update({'n_iter': n_iter, 'random_state': random_state})
        
        search = SearchClass(base_model, param_grid, **search_params)
        
        print(f"Starting {search_type} search with {cv}-fold CV...")
        search.fit(X_train_scaled, y_train)
        
        self.model = search.best_estimator_
        
        # Log best parameters
        wandb.config.update({"best_params": search.best_params_})
        wandb.log({"best_cv_score": search.best_score_})
        
        # Log CV results
        cv_results_df = pd.DataFrame(search.cv_results_)
        wandb.log({"cv_results": wandb.Table(dataframe=cv_results_df)})
        
        # Log metrics
        self._log_metrics(X_train_scaled, y_train, X_val_scaled, y_val, "tuned")
        
        # Plot hyperparameter importance
        self._plot_hyperparameter_importance(cv_results_df, param_grid)
        
        return self.model
    
    def train_with_feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        selection_method: str = "importance",
        n_features_to_select: Optional[int] = None,
        importance_threshold: float = 0.01,
        best_params: Optional[Dict] = None,
        random_state: int = 42,
    ) -> Tuple[xgb.XGBClassifier, List[str]]:
        """
        Entraîne XGBoost avec sélection de features.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_val: Features de validation
            y_val: Labels de validation
            selection_method: "importance", "shap", ou "recursive"
            n_features_to_select: Nombre de features à sélectionner (si None, utilise threshold)
            importance_threshold: Seuil pour feature importance
            best_params: Meilleurs hyperparamètres du tuning précédent
            random_state: Seed
            
        Returns:
            Tuple (modèle, liste des features sélectionnées)
        """
        config = {
            "model_type": "xgboost_feature_selection",
            "selection_method": selection_method,
            "n_features_initial": X_train.shape[1],
            "n_features_to_select": n_features_to_select,
            "importance_threshold": importance_threshold,
            "random_state": random_state,
        }
        
        if best_params:
            config["best_params"] = best_params
            
        self.init_wandb(config)
        
        # Standardiser les features pour l'entrainement initial
        if self.use_scaler:
            temp_scaler = StandardScaler()
            X_train_scaled_init = temp_scaler.fit_transform(X_train)
            X_val_scaled_init = temp_scaler.transform(X_val)
            X_train_scaled_init = pd.DataFrame(X_train_scaled_init, columns=X_train.columns, index=X_train.index)
            X_val_scaled_init = pd.DataFrame(X_val_scaled_init, columns=X_val.columns, index=X_val.index)
        else:
            X_train_scaled_init = X_train
            X_val_scaled_init = X_val
        
        # Entrainer un modele initial pour obtenir les importances
        if best_params:
            # Copier best_params et retirer random_state s'il existe pour eviter les doublons
            params_copy = best_params.copy()
            params_copy.pop('random_state', None)
            initial_model = xgb.XGBClassifier(**params_copy, random_state=random_state)
        else:
            initial_model = xgb.XGBClassifier(random_state=random_state)
            
        initial_model.fit(X_train_scaled_init, y_train)
        
        # Sélectionner les features
        if selection_method == "importance":
            selected_features = self._select_features_by_importance(
                initial_model,
                X_train.columns,
                n_features_to_select,
                importance_threshold
            )
        elif selection_method == "shap":
            try:
                import shap
                selected_features = self._select_features_by_shap(
                    initial_model,
                    X_train,
                    n_features_to_select,
                    importance_threshold
                )
            except ImportError:
                print("SHAP not installed, falling back to feature importance")
                selected_features = self._select_features_by_importance(
                    initial_model,
                    X_train.columns,
                    n_features_to_select,
                    importance_threshold
                )
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        wandb.config.update({
            "n_features_selected": len(selected_features),
            "selected_features": selected_features
        })
        
        # Entrainer le modele final avec les features selectionnees
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        
        # Reinitialiser et appliquer la standardisation sur les features selectionnees UNIQUEMENT
        if self.use_scaler:
            self.scaler = StandardScaler()  # Reinitialiser le scaler pour les features selectionnees
        X_train_scaled, X_val_scaled = self._scale_features(X_train_selected, X_val_selected, fit_scaler=True)
        
        if best_params:
            params_copy = best_params.copy()
            params_copy.pop('random_state', None)
            self.model = xgb.XGBClassifier(**params_copy, random_state=random_state)
        else:
            self.model = xgb.XGBClassifier(random_state=random_state)
            
        self.model.fit(X_train_scaled, y_train)
        
        # Log metrics (use scaled data)
        self._log_metrics(X_train_scaled, y_train, X_val_scaled, y_val, "feature_selected")
        
        # Plot feature importances
        self._plot_feature_importance(self.model, selected_features)
        
        return self.model, selected_features
    
    def _select_features_by_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        n_features: Optional[int],
        threshold: float
    ) -> List[str]:
        """Sélectionne les features par importance."""
        from ift6758.models.model_utils import plot_feature_importance
        
        importances = model.feature_importances_
        
        if n_features:
            # Sélectionner top N features
            indices = np.argsort(importances)[::-1][:n_features]
        else:
            # Sélectionner par seuil
            indices = np.where(importances >= threshold)[0]
            
        selected = [feature_names[i] for i in indices]
        
        # Log importance plot
        fig = plot_feature_importance(
            model,
            feature_names,
            title='Feature Importances (Selected Features)',
            top_n=len(selected)
        )
        wandb.log({"feature_selection/importance_plot": wandb.Image(fig)})
        plt.close()
        
        return selected
    
    def _select_features_by_shap(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        n_features: Optional[int],
        threshold: float
    ) -> List[str]:
        """Sélectionne les features par SHAP values."""
        import shap
        
        # Calculer SHAP values (sur un échantillon pour la vitesse)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except (ValueError, Exception) as e:
            print(f"Erreur SHAP TreeExplainer: {e}")
            print("Utilisation de feature importance à la place...")
            return self._select_features_by_importance(
                model, 
                X.columns.tolist(), 
                n_features, 
                threshold
            )
        
        # Calculer l'importance moyenne absolue
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        if n_features:
            indices = np.argsort(mean_abs_shap)[::-1][:n_features]
        else:
            max_shap = mean_abs_shap.max()
            indices = np.where(mean_abs_shap >= threshold * max_shap)[0]
            
        selected = [X.columns[i] for i in indices]
        
        # Plot SHAP summary
        fig, ax = plt.subplots(figsize=(10, max(6, len(selected) * 0.3)))
        shap.summary_plot(
            shap_values[:, indices],
            X_sample.iloc[:, indices],
            show=False
        )
        wandb.log({"feature_selection/shap_summary": wandb.Image(fig)})
        plt.close()
        
        return selected
    
    def _log_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        prefix: str
    ):
        """Log les métriques d'évaluation sur Wandb."""
        from ift6758.models.model_utils import log_metrics_to_wandb
        log_metrics_to_wandb(self.model, X_train, y_train, X_val, y_val, prefix, self.run)
    
    def _plot_hyperparameter_importance(
        self,
        cv_results: pd.DataFrame,
        param_grid: Dict
    ):
        """Plot l'importance des hyperparamètres."""
        # Pour chaque hyperparamètre, plot mean_test_score vs valeur
        n_params = len(param_grid)
        fig, axes = plt.subplots(
            nrows=(n_params + 1) // 2,
            ncols=2,
            figsize=(12, 4 * ((n_params + 1) // 2))
        )
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for idx, (param_name, param_values) in enumerate(param_grid.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Grouper par param et calculer mean score
            param_col = f'param_{param_name}'
            if param_col in cv_results.columns:
                grouped = cv_results.groupby(param_col)['mean_test_score'].mean()
                
                ax.plot(grouped.index.astype(str), grouped.values, marker='o')
                ax.set_xlabel(param_name)
                ax.set_ylabel('Mean CV Score')
                ax.set_title(f'Impact of {param_name}')
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        wandb.log({"hyperparameter_tuning/param_importance": wandb.Image(fig)})
        plt.close()
    
    def _plot_feature_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str]
    ):
        """Plot les importances des features."""
        from ift6758.models.model_utils import plot_feature_importance
        fig = plot_feature_importance(
            model, 
            feature_names,
            title="XGBoost Feature Importances",
            top_n=len(feature_names)
        )
        wandb.log({"model/feature_importance": wandb.Image(fig)})
        plt.close()
    
    def save_model(
        self,
        model_path: str,
        metadata: Optional[Dict] = None
    ):
        """Sauvegarde le modèle et le scaler localement et sur Wandb."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        joblib.dump(self.model, model_path)
        
        # Sauvegarder le scaler si utilise
        if self.use_scaler and self.scaler is not None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Modele et scaler sauvegardes dans {Path(model_path).parent}")
        else:
            print(f"Modele sauvegarde: {model_path}")

        # Sauvegarder sur Wandb
        if self.run:
            artifact = wandb.Artifact(
                name=f"model-{self.experiment_name}",
                type="model",
                metadata=metadata or {}
            )
            artifact.add_file(model_path)
            if self.use_scaler:
                artifact.add_file(scaler_path)
            self.run.log_artifact(artifact)
    
    def finish_run(self):
        """Termine la run Wandb."""
        if self.run:
            self.run.finish()
    
    @staticmethod
    def load_model(model_path: str, use_scaler: bool = True):
        """
        Charge un modele XGBoost sauvegarde avec son scaler.
        
        Args:
            model_path: Chemin vers le fichier .pkl du modele
            use_scaler: Si True, charge egalement le scaler
            
        Returns:
            Tuple (model, scaler) ou (model, None) si use_scaler=False
        """
        model = joblib.load(model_path)
        
        scaler = None
        if use_scaler:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            if Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Modele et scaler charges depuis {Path(model_path).parent}")
            else:
                print(f"Attention: Scaler non trouve a {scaler_path}")
                print(f"Le modele sera charge sans scaler")
        else:
            print(f"Modele charge: {model_path}")
        
        return model, scaler


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve


def generate_evaluation_plots(
    models_dict,
    X_val,
    y_val,
    output_dir: str = "./figures/milestone2"
):
    """
    Génère les 4 figures d'évaluation requises avec corrections de logique.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Préparer les prédictions
    predictions = {}
    for name, (model, features) in models_dict.items():
        X_subset = X_val[features].copy() if features else X_val.copy()
        predictions[name] = model.predict_proba(X_subset)[:, 1]

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - XGBoost Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Goal Rate vs Probability Percentile
    fig, ax = plt.subplots(figsize=(10, 8))
    n_bins = 20
    for name, y_pred_proba in predictions.items():
        df = pd.DataFrame({"prob": y_pred_proba, "goal": y_val})
        df["bin"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")
        mean_rates = df.groupby("bin", observed=True)["goal"].mean() * 100
        percentiles = np.linspace(0, 100, len(mean_rates))
        ax.plot(percentiles, mean_rates, marker="o", label=name, linewidth=2)

    ax.set_xlabel("Shot Probability Percentile", fontsize=12)
    ax.set_ylabel("Goal Rate (%)", fontsize=12)
    ax.set_title("Goal Rate vs Probability Percentile", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_goal_rate_percentile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Cumulative Proportion of Goals vs Probability Percentile
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_pred_proba in predictions.items():
        sorted_idx = np.argsort(y_pred_proba)[::-1]
        y_sorted = y_val.iloc[sorted_idx].values
        total_goals = np.sum(y_sorted)
        if total_goals == 0:
            continue
        cumsum = np.cumsum(y_sorted)
        cumsum_prop = cumsum / total_goals * 100
        n = len(y_sorted)
        percentiles_cum = np.arange(1, n + 1) / n * 100
        ax.plot(percentiles_cum, cumsum_prop, label=name, linewidth=2)

    ax.plot([0, 100], [0, 100], "k--", label="Perfect", linewidth=1)
    ax.set_xlabel("Percentile of Shots", fontsize=12)
    ax.set_ylabel("Cumulative Proportion of Goals (%)", fontsize=12)
    ax.set_title("Cumulative % of Goals vs Shot Probability Percentile",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_cumulative_goals.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_pred_proba in predictions.items():
        prob_true, prob_pred = calibration_curve(
            y_val, y_pred_proba, n_bins=10, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=name, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll evaluation plots saved to {output_dir}")


def load_and_prepare_data(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    target_col: str = "is_goal",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Charge et prépare les ensembles de données train/val/test.
    
    Args:
        train_path: Chemin vers train_data.csv (par défaut: data/processed/train_data.csv depuis la racine du projet)
        val_path: Chemin vers val_data.csv
        test_path: Chemin vers test_data.csv
        target_col: Nom de la colonne cible
        exclude_cols: Colonnes à exclure des features
    
    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Utiliser les chemins par défaut si non spécifiés
    if train_path is None or val_path is None or test_path is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data" / "processed"
        
        if train_path is None:
            train_path = str(data_dir / "train_data.csv")
        if val_path is None:
            val_path = str(data_dir / "val_data.csv")
        if test_path is None:
            test_path = str(data_dir / "test_data.csv")
    
    # Charger les données
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Colonnes à exclure par défaut (métadonnées)
    if exclude_cols is None:
        exclude_cols = [target_col, "season", "teamAbbr", "idGame", "prev_event", "prev_team"]
    
    # Identifier les features
    all_features = [col for col in train_df.columns if col not in exclude_cols]
    
    # Gérer les colonnes catégorielles (shot_type)
    categorical_cols = []
    for col in all_features:
        if train_df[col].dtype == 'object':
            categorical_cols.append(col)
            # One-hot encoding
            train_dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
            val_dummies = pd.get_dummies(val_df[col], prefix=col, drop_first=True)
            test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
            
            # Aligner les colonnes (au cas où certaines catégories manquent dans val/test)
            all_cols = sorted(set(train_dummies.columns) | set(val_dummies.columns) | set(test_dummies.columns))
            for dummy_col in all_cols:
                if dummy_col not in train_dummies:
                    train_dummies[dummy_col] = 0
                if dummy_col not in val_dummies:
                    val_dummies[dummy_col] = 0
                if dummy_col not in test_dummies:
                    test_dummies[dummy_col] = 0
            
            # Ajouter au dataframe et supprimer la colonne originale
            train_df = pd.concat([train_df, train_dummies[all_cols]], axis=1)
            val_df = pd.concat([val_df, val_dummies[all_cols]], axis=1)
            test_df = pd.concat([test_df, test_dummies[all_cols]], axis=1)
    
    # Mettre à jour la liste des features après encodage
    all_features = [col for col in train_df.columns if col not in exclude_cols and col not in categorical_cols]
    
    # Séparer X et y
    X_train = train_df[all_features]
    y_train = train_df[target_col]
    X_val = val_df[all_features]
    y_val = val_df[target_col]
    X_test = test_df[all_features]
    y_test = test_df[target_col]
    
    # Gérer les NaN
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            X_train[col].fillna(0, inplace=True)
            X_val[col].fillna(0, inplace=True)
            X_test[col].fillna(0, inplace=True)
    
    # Affichage simplifie
    print(f"Donnees chargees!")
    print(f"  Train: {X_train.shape[0]:,} tirs | {len(all_features)} features")
    print(f"  Val:   {X_val.shape[0]:,} tirs | Target: {y_train.mean():.2%} buts")
    print(f"  Test:  {X_test.shape[0]:,} tirs")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def perform_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    max_display: int = 20,
    sample_size: int = 1000,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Effectue une analyse SHAP complète sur un modèle XGBoost.
    
    Args:
        model: Modèle XGBoost entraîné
        X: Features (DataFrame)
        max_display: Nombre maximal de features à afficher
        sample_size: Taille de l'échantillon pour le calcul SHAP
        output_dir: Dossier de sortie pour sauvegarder les figures
        
    Returns:
        Dict contenant les SHAP values et les figures
    """
    try:
        import shap
    except ImportError:
        print("SHAP n'est pas installé. Installation: pip install shap")
        return {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Échantillonner pour la vitesse (SHAP peut être lent)
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    print(f"Calcul des SHAP values sur {len(X_sample)} échantillons...")
    
    # Utiliser shap.Explainer avec une fonction de prédiction
    # Plus robuste pour XGBoost récent
    print("Initialisation de l'explainer SHAP...")
    
    # Créer une fonction de prédiction qui retourne les probabilités
    def model_predict(X):
        return model.predict_proba(X)[:, 1]
    
    # Utiliser KernelExplainer ou Permutation pour compatibilité
    explainer = shap.KernelExplainer(model_predict, X_sample)
    shap_values = explainer.shap_values(X_sample)
    
    results = {
        'shap_values': shap_values,
        'base_value': explainer.expected_value,
        'X_sample': X_sample
    }
    
    # 1. Summary Plot (bar) - Importance globale
    print("\n Generating SHAP Summary Plot (bar)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'shap_importance_bar.png', dpi=150, bbox_inches='tight')
    results['importance_plot'] = fig
    plt.close()
    
    # 2. Summary Plot (beeswarm) - Distribution des impacts
    print("Generating SHAP Summary Plot (beeswarm)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Feature Impact Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
    results['summary_plot'] = fig
    plt.close()
    
    # 3. Dependence Plots pour top 4 features
    print("Generating SHAP Dependence Plots...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:4]
    top_features = [X_sample.columns[i] for i in top_features_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_features_idx):
        feat_name = X_sample.columns[feat_idx]
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_sample,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f'SHAP Dependence: {feat_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'shap_dependence_plots.png', dpi=150, bbox_inches='tight')
    results['dependence_plots'] = fig
    plt.close()
    
    # 4. Waterfall plot pour une prédiction exemple
    print("Generating SHAP Waterfall Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0].values,
            feature_names=X_sample.columns.tolist()
        ),
        max_display=15,
        show=False
    )
    plt.title('SHAP Waterfall Plot - Example Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'shap_waterfall.png', dpi=150, bbox_inches='tight')
    results['waterfall_plot'] = fig
    plt.close()
    
    # 5. Force plot (HTML interactive) pour une prédiction
    print("Generating SHAP Force Plot...")
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_sample.iloc[0],
        matplotlib=True,
        show=False
    )
    if output_dir:
        plt.savefig(output_dir / 'shap_force_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n SHAP analysis complete!")
    if output_dir:
        print(f"  Figures saved to {output_dir}")
    
    return results


def plot_correlation_matrix(
    X: pd.DataFrame,
    threshold: float = 0.7,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Affiche une matrice de corrélation pour identifier les features redondantes.
    
    Args:
        X: DataFrame des features
        threshold: Seuil pour identifier les corrélations fortes
        output_path: Chemin pour sauvegarder la figure
        
    Returns:
        Figure matplotlib
    """
    corr_matrix = X.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Masquer le triangle supérieur
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Identifier les paires fortement corrélées
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"\n Features fortement corrélées (|r| >= {threshold}):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   - {feat1} <-> {feat2}: r = {corr:.3f}")
        print("\n  Considérer la suppression d'une des features dans chaque paire")
    else:
        print(f"\n Aucune corrélation forte trouvée (threshold = {threshold})")
    
    return fig


def perform_recursive_feature_elimination(
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features_to_select: int = 10,
    step: int = 1,
    cv: int = 3,
    **model_kwargs
) -> Tuple[List[str], plt.Figure]:
    """
    Effectue une sélection de features par élimination récursive (RFE).
    
    Args:
        model_class: Classe du modèle (ex: xgb.XGBClassifier)
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        n_features_to_select: Nombre de features à sélectionner
        step: Nombre de features à retirer à chaque itération
        cv: Nombre de folds pour la cross-validation
        **model_kwargs: Paramètres du modèle
        
    Returns:
        Tuple (selected_features, figure)
    """
    from sklearn.feature_selection import RFE, RFECV
    
    print(f"\nRFE: Sélection de {n_features_to_select} features parmi {len(X_train.columns)}...")
    
    # Modèle de base
    estimator = model_class(**model_kwargs)
    
    # RFE avec cross-validation
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    rfecv.fit(X_train, y_train)
    
    # Features sélectionnées
    selected_features = X_train.columns[rfecv.support_].tolist()
    
    print(f" RFE completed")
    print(f"  Optimal number of features: {rfecv.n_features_}")
    print(f"  Selected features: {selected_features}")
    
    # Plot CV scores vs number of features
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
            rfecv.cv_results_['mean_test_score'],
            marker='o',
            linewidth=2,
            color='steelblue')
    ax.axvline(rfecv.n_features_, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {rfecv.n_features_} features')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cross-Validation Score (AUC)', fontsize=12)
    ax.set_title('RFE: CV Score vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return selected_features, fig


def perform_statistical_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 10,
    method: str = "mutual_info",
    output_path: Optional[str] = None
) -> Tuple[List[str], plt.Figure]:
    """
    Sélection de features par méthodes statistiques.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        n_features_to_select: Nombre de features à sélectionner
        method: "mutual_info" (Information Mutuelle) ou "chi2" (Chi-carré)
        output_path: Chemin pour sauvegarder la figure
        
    Returns:
        Tuple (selected_features, figure)
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
    
    print(f"\nSélection statistique ({method}): {n_features_to_select} features...")
    
    # Choisir la fonction de scoring
    if method == "mutual_info":
        score_func = mutual_info_classif
        method_name = "Mutual Information"
    elif method == "chi2":
        # Chi2 nécessite des valeurs non-négatives
        # Normaliser pour avoir des valeurs >= 0
        X_normalized = X_train.copy()
        for col in X_normalized.columns:
            X_normalized[col] = X_normalized[col] - X_normalized[col].min()
        score_func = chi2
        method_name = "Chi-Squared"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sélection
    if method == "chi2":
        selector = SelectKBest(score_func=score_func, k=n_features_to_select)
        selector.fit(X_normalized, y_train)
    else:
        selector = SelectKBest(score_func=score_func, k=n_features_to_select)
        selector.fit(X_train, y_train)
    
    # Features sélectionnées
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Scores de toutes les features
    scores = selector.scores_
    
    print(f"✓ Sélection complétée")
    print(f"  Features sélectionnées: {selected_features}")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Trier par score décroissant
    sorted_idx = np.argsort(scores)[::-1]
    sorted_features = [X_train.columns[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    # Couleurs: vert pour sélectionnées, bleu pour les autres
    colors = ['green' if feat in selected_features else 'steelblue' 
              for feat in sorted_features]
    
    ax.barh(range(len(sorted_features)), sorted_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_xlabel(f'{method_name} Score', fontsize=12)
    ax.set_title(f'Feature Selection: {method_name}', fontsize=14, fontweight='bold')
    ax.axvline(scores[sorted_idx[n_features_to_select-1]], color='red', 
               linestyle='--', linewidth=2, label=f'Top {n_features_to_select} threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return selected_features, fig


def perform_sequential_feature_selection(
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features_to_select: int = 10,
    direction: str = "forward",
    cv: int = 3,
    **model_kwargs
) -> Tuple[List[str], plt.Figure]:
    """
    Sélection séquentielle de features (Forward ou Backward Selection).
    Méthode Wrapper qui évalue des sous-ensembles de features.
    
    Args:
        model_class: Classe du modèle
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        n_features_to_select: Nombre de features à sélectionner
        direction: "forward" (ajoute progressivement) ou "backward" (retire progressivement)
        cv: Nombre de folds pour la cross-validation
        **model_kwargs: Paramètres du modèle
        
    Returns:
        Tuple (selected_features, figure)
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    
    print(f"\nSequential Feature Selection ({direction}): {n_features_to_select} features...")
    
    estimator = model_class(**model_kwargs)
    
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    
    sfs.fit(X_train, y_train)
    
    selected_features = X_train.columns[sfs.get_support()].tolist()
    
    print(f"SFS completed")
    print(f"Selected features: {selected_features}")
    
    # Évaluer la performance progressive
    scores = []
    n_features_range = range(1, len(X_train.columns) + 1) if direction == "forward" else range(len(X_train.columns), 0, -1)
    
    # Simuler l'ajout/retrait progressif (approximation pour visualisation)
    feature_names = []
    for n in list(n_features_range)[:15]:  # Limiter à 15 pour la vitesse
        if n <= len(selected_features):
            temp_features = selected_features[:n]
            estimator_temp = model_class(**model_kwargs)
            estimator_temp.fit(X_train[temp_features], y_train)
            score = roc_auc_score(y_val, estimator_temp.predict_proba(X_val[temp_features])[:, 1])
            scores.append(score)
            feature_names.append(n)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(feature_names, scores, marker='o', linewidth=2, color='green')
    ax.axvline(n_features_to_select, color='red', linestyle='--', linewidth=2,
               label=f'Selected: {n_features_to_select} features')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title(f'Sequential Feature Selection ({direction.capitalize()})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return selected_features, fig


def perform_l1_regularization_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features_to_select: int = 10,
    output_path: Optional[str] = None
) -> Tuple[List[str], plt.Figure]:
    """
    Sélection de features par régularisation L1 (Lasso).
    Méthode embarquée qui sélectionne via pénalisation.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        n_features_to_select: Nombre de features à sélectionner
        output_path: Chemin pour sauvegarder la figure
        
    Returns:
        Tuple (selected_features, figure)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    print(f"\nL1 Regularization Selection: {n_features_to_select} features...")
    
    # Standardiser (nécessaire pour L1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Tester différentes valeurs de C (inverse de la régularisation)
    C_values = np.logspace(-3, 2, 50)
    n_features_per_C = []
    auc_scores = []
    
    for C in C_values:
        model = LogisticRegression(
            penalty='l1',
            C=C,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        
        # Nombre de features non-nulles
        n_features = np.sum(model.coef_ != 0)
        n_features_per_C.append(n_features)
        
        # Score AUC
        if n_features > 0:
            auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
        else:
            auc = 0.5
        auc_scores.append(auc)
    
    # Trouver le C qui donne environ n_features_to_select
    closest_idx = np.argmin(np.abs(np.array(n_features_per_C) - n_features_to_select))
    optimal_C = C_values[closest_idx]
    
    # Entraîner avec le C optimal
    model_final = LogisticRegression(
        penalty='l1',
        C=optimal_C,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    model_final.fit(X_train_scaled, y_train)
    
    # Features sélectionnées
    selected_mask = model_final.coef_[0] != 0
    selected_features = X_train.columns[selected_mask].tolist()
    
    print(f"  L1 Selection completed")
    print(f"  Optimal C: {optimal_C:.4f}")
    print(f"  Features sélectionnées ({len(selected_features)}): {selected_features}")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Nombre de features vs C
    ax1.semilogx(C_values, n_features_per_C, 'b-', linewidth=2)
    ax1.axhline(n_features_to_select, color='red', linestyle='--', linewidth=2,
                label=f'Target: {n_features_to_select} features')
    ax1.axvline(optimal_C, color='green', linestyle='--', linewidth=2,
                label=f'Optimal C: {optimal_C:.4f}')
    ax1.set_xlabel('C (Inverse of Regularization)', fontsize=12)
    ax1.set_ylabel('Number of Non-Zero Features', fontsize=12)
    ax1.set_title('L1 Regularization: Feature Sparsity', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC vs Nombre de features
    ax2.plot(n_features_per_C, auc_scores, 'g-', linewidth=2)
    ax2.axvline(n_features_per_C[closest_idx], color='red', linestyle='--', linewidth=2,
                label=f'Selected: {len(selected_features)} features')
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Validation AUC', fontsize=12)
    ax2.set_title('Performance vs Model Complexity', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return selected_features, fig


def compare_feature_selection_methods(
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features: int = 10,
    output_dir: Optional[str] = None,
    **model_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare toutes les méthodes de sélection de features.
    
    Args:
        model_class: Classe du modèle (ex: xgb.XGBClassifier)
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        n_features: Nombre de features à sélectionner
        output_dir: Dossier pour sauvegarder les figures
        **model_kwargs: Paramètres du modèle
        
    Returns:
        Dict contenant les résultats de chaque méthode
    """
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*80)
    print("COMPARAISON DES MÉTHODES DE SÉLECTION DE FEATURES")
    print("="*80)
    
    results = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance (Embarquée - XGBoost)
    print("\n MÉTHODE EMBARQUÉE: Feature Importance XGBoost")
    model_temp = model_class(**model_kwargs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model_temp.fit(X_train_scaled, y_train)
    
    importances = model_temp.feature_importances_
    top_indices = np.argsort(importances)[::-1][:n_features]
    features_importance = X_train.columns[top_indices].tolist()
    
    X_val_subset = X_val_scaled[:, top_indices]
    auc_importance = roc_auc_score(y_val, model_temp.predict_proba(X_val_scaled)[:, 1])
    
    results['Feature Importance'] = {
        'features': features_importance,
        'n_features': len(features_importance),
        'auc': auc_importance,
        'method_type': 'Embarquée'
    }
    
    # 2. Mutual Information (Statistique)
    print("\n MÉTHODE STATISTIQUE: Mutual Information")
    features_mi, _ = perform_statistical_feature_selection(
        X_train, y_train, n_features_to_select=n_features, method="mutual_info",
        output_path=str(output_dir / "q3_statistical_mutual_info.png") if output_dir else None
    )
    
    # Évaluer
    model_mi = model_class(**model_kwargs)
    X_train_mi = scaler.fit_transform(X_train[features_mi])
    X_val_mi = scaler.transform(X_val[features_mi])
    model_mi.fit(X_train_mi, y_train)
    auc_mi = roc_auc_score(y_val, model_mi.predict_proba(X_val_mi)[:, 1])
    
    results['Mutual Information'] = {
        'features': features_mi,
        'n_features': len(features_mi),
        'auc': auc_mi,
        'method_type': 'Statistique'
    }
    
    # 3. RFE (Wrapper)
    print("\n MÉTHODE WRAPPER: Recursive Feature Elimination")
    features_rfe, _ = perform_recursive_feature_elimination(
        model_class, X_train, y_train, X_val, y_val,
        n_features_to_select=n_features, cv=3, **model_kwargs
    )
    
    X_train_rfe = scaler.fit_transform(X_train[features_rfe])
    X_val_rfe = scaler.transform(X_val[features_rfe])
    model_rfe = model_class(**model_kwargs)
    model_rfe.fit(X_train_rfe, y_train)
    auc_rfe = roc_auc_score(y_val, model_rfe.predict_proba(X_val_rfe)[:, 1])
    
    results['RFE'] = {
        'features': features_rfe,
        'n_features': len(features_rfe),
        'auc': auc_rfe,
        'method_type': 'Wrapper'
    }
    
    # 4. L1 Regularization (Embarquée)
    print("\n MÉTHODE EMBARQUÉE: L1 Regularization")
    features_l1, _ = perform_l1_regularization_selection(
        X_train, y_train, X_val, y_val, n_features_to_select=n_features,
        output_path=str(output_dir / "q3_l1_regularization.png") if output_dir else None
    )
    
    X_train_l1 = scaler.fit_transform(X_train[features_l1])
    X_val_l1 = scaler.transform(X_val[features_l1])
    model_l1 = model_class(**model_kwargs)
    model_l1.fit(X_train_l1, y_train)
    auc_l1 = roc_auc_score(y_val, model_l1.predict_proba(X_val_l1)[:, 1])
    
    results['L1 Regularization'] = {
        'features': features_l1,
        'n_features': len(features_l1),
        'auc': auc_l1,
        'method_type': 'Embarquée'
    }
    
    # Comparaison visuelle
    print("\n" + "="*80)
    print("RÉSUMÉ COMPARATIF")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC par méthode
    methods = list(results.keys())
    aucs = [results[m]['auc'] for m in methods]
    colors_map = {'Embarquée': 'steelblue', 'Wrapper': 'green', 
              'Statistique': 'orange', 'Interprétabilité': 'purple'}
    bar_colors = [colors_map[results[m]['method_type']] for m in methods]
    
    ax1.barh(methods, aucs, color=bar_colors, alpha=0.7)
    ax1.set_xlabel('Validation AUC', fontsize=12)
    ax1.set_title('Performance par Méthode de Sélection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, (method, auc) in enumerate(zip(methods, aucs)):
        ax1.text(auc + 0.001, i, f'{auc:.4f}', va='center', fontsize=10)
    
    # Plot 2: Chevauchement des features
    all_features_sets = [set(results[m]['features']) for m in methods]
    
    # Compter les chevauchements
    overlap_matrix = np.zeros((len(methods), len(methods)))
    for i, set_i in enumerate(all_features_sets):
        for j, set_j in enumerate(all_features_sets):
            overlap_matrix[i, j] = len(set_i & set_j)
    
    im = ax2.imshow(overlap_matrix, cmap='YlGn', aspect='auto')
    ax2.set_xticks(range(len(methods)))
    ax2.set_yticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.set_title('Chevauchement des Features Sélectionnées', fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(methods)):
        for j in range(len(methods)):
            text = ax2.text(j, i, int(overlap_matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax2, label='Nombre de features en commun')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "q3_methods_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Afficher le tableau récapitulatif
    print(f"\n{'Méthode':<25} {'Type':<15} {'AUC':<10} {'Features'}")
    print("-" * 80)
    for method, data in results.items():
        print(f"{method:<25} {data['method_type']:<15} {data['auc']:<10.4f} {data['n_features']}")
    
    # Features communes
    common_features = set.intersection(*all_features_sets)
    if common_features:
        print(f"\n Features communes à TOUTES les méthodes ({len(common_features)}):")
        for feat in common_features:
            print(f"  - {feat}")
    else:
        print("\n Aucune feature commune à toutes les méthodes")
    
    # Recommandation
    best_method = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n RECOMMANDATION: '{best_method[0]}' (AUC = {best_method[1]['auc']:.4f})")
    
    return results
