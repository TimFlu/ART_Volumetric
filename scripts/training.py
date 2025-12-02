import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from itertools import product
import warnings
import pickle as pkl
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.optimize import dual_annealing
from scipy.stats import wilcoxon
import logging
from datetime import datetime

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def configure_logging(logger, output_path):
    """Configures logging for a new dataset by creating a new log file."""
    while logger.hasHandlers():
        logger.handlers.clear()

    log_file = os.path.join(output_path, f"{current_time}_training.log")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Logging configured for: {log_file}")
    print(f"Logging to {log_file}")
    return logger

def extract_features(volumes, variant='A', last_n=5):
    """Extract features with additional variants"""
    x_all = np.arange(len(volumes)).reshape(-1, 1)
    y_all = np.array(volumes).reshape(-1, 1)

    slope_all = LinearRegression().fit(x_all, y_all).coef_[0][0]
    slope_last = LinearRegression().fit(x_all[-last_n:], y_all[-last_n:]).coef_[0][0]
    mean_last_two = np.mean(volumes[-2:])
    std_all = np.std(volumes)
    min_val = np.min(volumes)
    max_val = np.max(volumes)
    range_val = max_val - min_val
    
    features = {
        'A': [mean_last_two, slope_last, slope_all],
        'B': [mean_last_two, slope_last],
        'C': [mean_last_two, slope_all],
        'D': [mean_last_two],
        'E': [mean_last_two, slope_last, slope_all, std_all],  # Added std
        'F': [mean_last_two, slope_last, slope_all, range_val],  # Added range
        'G': [mean_last_two, slope_last, slope_all, min_val, max_val],  # Added extremes
    }
    return features[variant]

class ModelEvaluator:
    """Comprehensive model evaluation with clinical metrics"""
    
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        
    def calculate_clinical_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        tp = fp = fn = tn = 0
        early_detections = []
        late_detections = []
        
        for yt, yp in zip(y_true, y_pred):
            if yt > 0:  # Patient needs replanning
                if yp > 0:  # Model predicts replanning
                    if yp <= yt + self.tolerance:  # Within tolerance
                        tp += 1
                        if yp < yt:
                            early_detections.append(yt - yp)
                        elif yp > yt:
                            late_detections.append(yp - yt)
                    else:  # Too late
                        fn += 1
                        late_detections.append(yp - yt)
                else:  # Missed replanning
                    fn += 1
            else:  # Patient doesn't need replanning
                if yp > 0:  # False alarm
                    fp += 1
                else:  # Correct non-replanning
                    tn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        
        # Clinical timing metrics
        avg_early_detection = np.mean(early_detections) if early_detections else 0
        avg_late_detection = np.mean(late_detections) if late_detections else 0
        early_detection_rate = len(early_detections) / len(y_true) if len(y_true) > 0 else 0
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'avg_early_detection': avg_early_detection,
            'avg_late_detection': avg_late_detection,
            'early_detection_rate': early_detection_rate,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def calculate_composite_score(self, metrics, weights=None):
        """Calculate weighted composite score for model ranking"""
        if weights is None:
            weights = {
                'f1_score': 0.3,
                'recall': 0.5,
                'specificity': 0.1,
                'early_detection_rate': 0.1
            }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
        
        return score

class ThresholdModel(BaseEstimator, ClassifierMixin):
    def __init__(self, custom_loss_fn, search_grid=None, min_t=5, max_t=50,
                 alpha_late=2.0, beta_fn=20.0, gamma_fp=1.0):
        self.thresholds = None
        self.search_grid = search_grid if search_grid is not None else np.linspace(-6, 0, 50)
        self.min_t = min_t
        self.max_t = max_t
        self.alpha_late = alpha_late
        self.beta_fn = beta_fn
        self.gamma_fp = gamma_fp
        self.custom_loss_fn = custom_loss_fn

    def fit(self, X_seqs, y_true):
        """Fit with improved optimization"""
        if len(X_seqs[0][0]) == 0:
            raise ValueError("Empty feature vectors")
        
        self.best_iter = None
        self.best_value = np.inf
        self.current_iter = 0
        def callback(x, f, context):
            # iteration counter
            self.current_iter += 1
            # track best f
            if f < self.best_value:
                self.best_value = f
                self.best_iter = self.current_iter
                    
        bounds = [(-6, 0) for _ in range(len(X_seqs[0][0]))]
        
        # Multiple random starts for better optimization
        best_loss = float('inf')
        best_thresh = None
        
        for _ in range(3):  # Multiple starts
            result = dual_annealing(
                lambda x: self.custom_loss_fn(y_true, [self._predict_single(x_seq, x) for x_seq in X_seqs], 
                                            self.alpha_late, self.beta_fn, self.gamma_fp),
                bounds=bounds, 
                maxiter=300,
                seed=np.random.randint(0, 1000),
                # callback=callback
            )
            # logger.warning(f"Optimization finished after {self.current_iter} iterations with loss {result.fun} and thresholds {result.x}")
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_thresh = result.x

        self.thresholds = np.array(best_thresh)
        return self

    def _predict_single(self, x_seq, thresholds):
        """Returns the earliest time t where all features < thresholds"""
        for t, x in enumerate(x_seq, start=self.min_t):
            if (np.array(x) < thresholds).all():
                return t
        return 0

    def predict(self, X_seqs):
        return [self._predict_single(x_seq, self.thresholds) for x_seq in X_seqs]

def custom_loss(y_true, y_pred, alpha_late, beta_fn, gamma_fp):
    total = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            if yp > 0:
                total += gamma_fp  # False positive
        else:
            if yp == 0:
                total += beta_fn  # False negative
            elif yp > yt:
                total += alpha_late * (yp - yt)  # Late replanning (linear penalty)
            else:
                total += 0.5 * (yt - yp)  # Early replanning (smaller penalty)
    return total

def create_evaluation_plots(summary_df, output_path):
    """Create comprehensive evaluation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # F1 Score by variant
    variant_f1 = summary_df.groupby('variant')['f1_outer'].agg(['mean', 'std'])
    axes[0, 0].bar(variant_f1.index, variant_f1['mean'], yerr=variant_f1['std'])
    axes[0, 0].set_title('F1 Score by Feature Variant')
    axes[0, 0].set_ylabel('F1 Score')
    
    # Balanced Accuracy by variant
    variant_ba = summary_df.groupby('variant')['ba_outer'].agg(['mean', 'std'])
    axes[0, 1].bar(variant_ba.index, variant_ba['mean'], yerr=variant_ba['std'])
    axes[0, 1].set_title('Balanced Accuracy by Feature Variant')
    axes[0, 1].set_ylabel('Balanced Accuracy')
    
    # Average Detection Time
    variant_det = summary_df.groupby('variant')['avg_det'].agg(['mean', 'std'])
    axes[0, 2].bar(variant_det.index, variant_det['mean'], yerr=variant_det['std'])
    axes[0, 2].set_title('Average Detection Time by Variant')
    axes[0, 2].set_ylabel('Detection Time (Fractions)')
    
    # Hyperparameter analysis
    if 'alpha_late' in summary_df.columns:
        axes[1, 0].scatter(summary_df['alpha_late'], summary_df['f1_outer'], alpha=0.6)
        axes[1, 0].set_xlabel('Alpha Late')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 vs Alpha Late Parameter')
    
    # Parameter correlations
    param_cols = [col for col in summary_df.columns if 'param' in col.lower()]
    if len(param_cols) > 0:
        corr_data = summary_df[['f1_outer', 'ba_outer'] + param_cols].corr()
        im = axes[1, 1].imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(corr_data.columns)))
        axes[1, 1].set_xticklabels(corr_data.columns, rotation=45)
        axes[1, 1].set_yticks(range(len(corr_data.columns)))
        axes[1, 1].set_yticklabels(corr_data.columns)
        axes[1, 1].set_title('Parameter Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 1])
    
    # Performance distribution
    axes[1, 2].boxplot([summary_df[summary_df['variant'] == v]['f1_outer'].values 
                       for v in summary_df['variant'].unique()],
                      labels=summary_df['variant'].unique())
    axes[1, 2].set_title('F1 Score Distribution by Variant')
    axes[1, 2].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{current_time}_comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the data
    DATA_PATH = './data'
    all_data = pkl.load(open(os.path.join(DATA_PATH, 'all_data.pkl'), 'rb'))
    
    # Remove data of CT and rCT scans
    for patient_id in all_data.keys():
        all_data[patient_id]['data'] = all_data[patient_id]['data'].drop([
            index for index in all_data[patient_id]['data'].index 
            if 'REPLAN' in index or 'RT' in index or 'PT' in index
        ])

    # Prepare patient data
    patients = [{
        'patient_id': int(patient_id.lstrip('0'))-1,
        'volume': all_data[patient_id]['data']['volume difference to pCT [%]'].values,
        'label': all_data[patient_id]['label'][0]
    } for patient_id in all_data.keys()]
    logger.info(f'Total patients loaded: {len(patients)}')

    # Summarize dataset
    for patient_id, label, fractions in zip([p["patient_id"] for p in patients], [p["label"] for p in patients],[len(p["volume"]) for p in patients]):
        logger.info(f'Patient {patient_id}: Label={label}, Fractions={fractions}')


    # Expanded feature variants and hyperparameters
    feature_variants = ['A', 'B', 'C', 'D']
    # More comprehensive hyperparameter grid
    loss_params_grid = list(product(
        [0.5, 1.0, 1.5, 2.0],  # alpha_late
        [225, 300, 450, 600],  # beta_fn  
        [150, 200, 300, 400]        # gamma_fp
    ))
    # loss_params_grid = list(product(
    #     [0.5, 1.0, 1.5, 2.0],  # alpha_late
    #     [600],  # beta_fn  
    #     [150, 200, 300, 400]        # gamma_fp
    # ))
    
    # Filter invalid combinations
    loss_params_grid = [(a, b, g) for a, b, g in loss_params_grid if b > g]


    # Use Leave-One-Out CV for better evaluation with small dataset
    # ----------------- SINGLE LOOCV MODEL SELECTION -----------------
    loo = LeaveOneOut()
    y_all = [p['label'] for p in patients]

    combos = [(v, a, b, g) for v in feature_variants
                            for (a, b, g) in loss_params_grid]

    evaluator = ModelEvaluator(tolerance=5)

    results = []  # one row per combo
    all_fold_preds = {}  # optional: store per-fold preds for later analysis


    logger.info(f"Loss parameters are {loss_params_grid}")
    logger.info(f"Feature variants are {feature_variants}")
    logger.info('----------------------------------------')
    logger.info(f'Starting LOOCV model selection over {len(combos)} combinations...')

    for variant, alpha_late, beta_fn, gamma_fp in combos:
        logger.info(f"Evaluating variant={variant}, α={alpha_late}, β={beta_fn}, γ={gamma_fp}")
        fold_preds = []
        fold_truth = []
        
        for train_idx, test_idx in loo.split(np.zeros(len(y_all)), y_all):
            # logger.info(f"Training on {len(train_idx)} patients, testing on patient {test_idx[0]}")
            train_patients = [patients[i] for i in train_idx]
            test_patient   = patients[test_idx[0]]

            # Build training sequences
            X_train = [[extract_features(p['volume'][:t], variant)
                        for t in range(5, 36)] for p in train_patients]
            y_train = [p['label'] for p in train_patients]

            # Fit thresholds on the 20 training patients
            model = ThresholdModel(custom_loss_fn=custom_loss,
                                alpha_late=alpha_late,
                                beta_fn=beta_fn,
                                gamma_fp=gamma_fp)
            model.fit(X_train, y_train)

            # Predict left-out patient
            X_test = [[extract_features(test_patient['volume'][:t], variant)
                    for t in range(5, 36)]]
            y_pred = model.predict(X_test)[0]

            fold_preds.append(y_pred)
            fold_truth.append(test_patient['label'])
        
        # Compute metrics for this combo
        metrics = evaluator.calculate_clinical_metrics(fold_truth, fold_preds)
        comp_score = evaluator.calculate_composite_score(metrics,)
        results.append({
            'variant': variant,
            'alpha_late': alpha_late,
            'beta_fn': beta_fn,
            'gamma_fp': gamma_fp,
            'f1': metrics['f1_score'],
            'balanced_acc': metrics['balanced_accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'specificity': metrics['specificity'],
            'fpr': metrics['false_positive_rate'],
            'fnr': metrics['false_negative_rate'],
            'avg_early_detection': metrics['avg_early_detection'],
            'avg_late_detection': metrics['avg_late_detection'],
            'early_detection_rate': metrics['early_detection_rate'],
            'composite': comp_score
        })
        all_fold_preds[(variant, alpha_late, beta_fn, gamma_fp)] = (fold_truth, fold_preds)

    logger.info(f'all fold preds: {all_fold_preds}')
    pd.to_pickle(all_fold_preds, os.path.join(output_path, f'loo_all_fold_preds.pkl'))
    # ----------------- SELECT BEST COMBO -----------------
    results_df = pd.DataFrame(results)
    best_idx = results_df['composite'].idxmax()  # or 'f1'
    best_row = results_df.loc[best_idx]
    best_variant = best_row['variant']
    best_alpha   = best_row['alpha_late']
    best_beta    = best_row['beta_fn']
    best_gamma   = best_row['gamma_fp']
    

    logger.info(f"BEST: variant={best_variant}, α={best_alpha}, β={best_beta}, γ={best_gamma}, "
                f"F1={best_row['f1']:.3f}, composite={best_row['composite']:.3f}")

    # ----------------- FINAL REFIT ON ALL DATA -----------------
    X_full = [[extract_features(p['volume'][:t], best_variant) for t in range(5, 36)]
            for p in patients]
    y_full = [p['label'] for p in patients]

    final_model = ThresholdModel(custom_loss_fn=custom_loss,
                                alpha_late=best_alpha,
                                beta_fn=best_beta,
                                gamma_fp=best_gamma)
    final_model.fit(X_full, y_full)
    logger.info(f"Final thresholds: {final_model.thresholds}")

    # Optional: evaluate on training data just for sanity (not to claim generalisation)
    final_preds = final_model.predict(X_full)
    final_metrics = evaluator.calculate_clinical_metrics(y_full, final_preds)
    for k, v in final_metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")

    # Save results
    save_folder = os.path.join(output_path, 'plots')
    os.makedirs(save_folder, exist_ok=True)
    results_df.to_csv(os.path.join(save_folder, f'../{current_time}_loo_selection_results.csv'), index=False)


    # Create comprehensive plots
    # create_evaluation_plots(results_df, save_folder)
    
    # Individual patient plots
    for patient, pred in zip(patients, final_preds):
        volumes = patient['volume']
        patient_id = patient['patient_id']
        labels = patient['label']
        
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(1, len(volumes)+1), volumes, 'b.', label=f'Patient {patient_id}')
        plt.title(f'Patient {patient_id} - True: {labels}, Predicted: {pred}')
        plt.xlabel('Fraction')
        plt.ylabel('Volume difference to pCT [%]')
        
        if labels > 0:
            plt.axvline(x=labels, color='g', linestyle='-', label='Actual Replanning')
        if pred > 0:
            plt.axvline(x=pred, color='r', linestyle='--', label='Predicted Replanning')
            
        plt.xticks(np.arange(0, len(volumes)+1, 1))
        plt.ylim(-14, 7)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(save_folder, f'patient_{patient_id}_final.png'), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f'All results saved in {save_folder}')

if __name__ == '__main__':
    np.random.seed(42)
    logger = logging.getLogger(__name__)
    output_path = './results/'
    output_path = os.path.join(output_path, f'{current_time}')
    os.makedirs(output_path, exist_ok=True)
    
    logger = configure_logging(logger, output_path)
    main()