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
        project_name: str = "ift6758-milestone2",
        experiment_name: str = "xgboost-baseline",
        wandb_entity: Optional[str] = "IFT6758.2025-A04",
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
        random_state: int = 42,
    ) -> xgb.XGBClassifier:
        """
        Entraîne un XGBoost baseline avec distance et angle uniquement.
            
        Returns:
            Modèle XGBoost entraîné
        """
        config = {
            "model_type": "xgboost_baseline",
            "features": features,
            "n_features": len(features),
            "random_state": random_state,
            "use_scaler": self.use_scaler,
            "hyperparameters": "default"
        }
        
        self.init_wandb(config)
        
        # Sélectionner et standardiser les features
        X_train_subset = X_train[features]
        X_val_subset = X_val[features]
        X_train_scaled, X_val_scaled = self._scale_features(X_train_subset, X_val_subset, fit_scaler=True)
        
        # Entraîner avec paramètres par défaut
        self.model = xgb.XGBClassifier(
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
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


def generate_evaluation_plots(
    models_dict: Dict[str, Tuple[Any, str]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str = "./figures/milestone2"
):
    """
    Génère les 4 figures d'évaluation requises.
    
    Args:
        models_dict: Dict {nom_modèle: (modèle, features ou None si toutes)}
        X_val: Features de validation
        y_val: Labels de validation
        output_dir: Dossier de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les données pour chaque modèle
    predictions = {}
    for name, (model, features) in models_dict.items():
        if features:
            X_subset = X_val[features]
        else:
            X_subset = X_val
        predictions[name] = model.predict_proba(X_subset)[:, 1]
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - XGBoost Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Goal Rate vs Probability Percentile
    fig, ax = plt.subplots(figsize=(10, 8))
    percentiles = np.linspace(0, 100, 21)
    
    for name, y_pred_proba in predictions.items():
        goal_rates = []
        for p in percentiles:
            threshold = np.percentile(y_pred_proba, p)
            mask = y_pred_proba >= threshold
            if mask.sum() > 0:
                goal_rate = y_val[mask].mean()
                goal_rates.append(goal_rate * 100)
            else:
                goal_rates.append(0)
        ax.plot(percentiles, goal_rates, marker='o', label=name, linewidth=2)
    
    ax.set_xlabel('Shot Probability Percentile', fontsize=12)
    ax.set_ylabel('Goal Rate (%)', fontsize=12)
    ax.set_title('Goal Rate vs Probability Percentile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_goal_rate_percentile.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Cumulative Proportion of Goals vs Percentile
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, y_pred_proba in predictions.items():
        # Trier par probabilité décroissante
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_sorted = y_val.iloc[sorted_indices].values
        
        # Proportion cumulée
        cumsum = np.cumsum(y_sorted)
        cumsum_prop = cumsum / cumsum[-1] * 100
        
        # Percentiles
        n = len(y_sorted)
        percentiles_cum = np.arange(1, n + 1) / n * 100
        
        ax.plot(percentiles_cum, cumsum_prop, label=name, linewidth=2)
    
    ax.plot([0, 100], [0, 100], 'k--', label='Perfect', linewidth=1)
    ax.set_xlabel('Percentile of Shots', fontsize=12)
    ax.set_ylabel('Cumulative Proportion of Goals (%)', fontsize=12)
    ax.set_title('Cumulative % of Goals vs Shot Probability Percentile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_cumulative_goals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, y_pred_proba in predictions.items():
        prob_true, prob_pred = calibration_curve(
            y_val,
            y_pred_proba,
            n_bins=10,
            strategy='uniform'
        )
        ax.plot(prob_pred, prob_true, marker='o', label=name, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1)
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_calibration_curves.png', dpi=150, bbox_inches='tight')
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
