import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

from src.model import CLASSES, build_model
from src.dataset import SoyLeafDataset, get_eval_transforms

def main():
    torch.set_num_threads(8)
    df = pd.read_csv('data/splits_5fold.csv')
    img_dir = 'data/p4-soy-leaf-health/images'

    model = build_model(pretrained=False, checkpoint_path='models/mobilenet_v3.pt')
    model.eval()
    transform = get_eval_transforms()

    os.makedirs('reports', exist_ok=True)

    fold_metrics = []
    all_cv_preds = []
    all_cv_targets = []
    all_cv_probs = []
    cv_records = []

    cv_folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

    print('Evaluating baseline model on 5 CV folds using batched TTA...', flush=True)

    with torch.no_grad():
        for fold in cv_folds:
            f_df = df[df['split'] == fold].reset_index(drop=True)
            ds = SoyLeafDataset(f_df, img_dir, transform=transform)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

            fold_preds = []
            fold_targets = []
            fold_probs = []

            for imgs, targets, img_ids in loader:
                p = model.predict_tta(imgs).cpu().numpy()
                preds = np.argmax(p, axis=1)

                fold_preds.extend(preds)
                fold_targets.extend(targets.numpy())
                fold_probs.append(p)

                for b in range(len(img_ids)):
                    pred_cls = CLASSES[preds[b]]
                    true_cls = CLASSES[targets[b]]
                    prob_vec = p[b]
                    cv_records.append({
                        'image_id': img_ids[b],
                        'fold': fold,
                        'true_label': true_cls,
                        'pred_label': pred_cls,
                        'true_prob': float(prob_vec[targets[b]]),
                        'pred_prob': float(prob_vec[preds[b]]),
                        'loss': float(-np.log(max(prob_vec[targets[b]], 1e-7))),
                        'is_error': int(pred_cls != true_cls)
                    })

            acc = accuracy_score(fold_targets, fold_preds)
            mf1 = f1_score(fold_targets, fold_preds, average='macro', zero_division=0)
            fold_metrics.append({'fold': fold, 'accuracy': acc, 'macro_f1': mf1})
            print(f'{fold} -> Accuracy: {acc*100:.2f}%, Macro-F1: {mf1:.4f}', flush=True)

            all_cv_preds.extend(fold_preds)
            all_cv_targets.extend(fold_targets)
            all_cv_probs.extend(np.vstack(fold_probs))

    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['macro_f1'] for m in fold_metrics]
    print('=' * 65, flush=True)
    print(f'5-Fold CV Mean Accuracy: {np.mean(accs)*100:.2f}% +- {np.std(accs)*100:.2f}%', flush=True)
    print(f'5-Fold CV Mean Macro-F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}', flush=True)
    print('=' * 65, flush=True)

    # Classification report
    print('\n--- Classification Report (All 1,784 CV samples) ---', flush=True)
    report_dict = classification_report(all_cv_targets, all_cv_preds, target_names=CLASSES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv('reports/baseline_classification_report.csv')
    print(classification_report(all_cv_targets, all_cv_preds, target_names=CLASSES, digits=4), flush=True)

    # Confusion matrix
    cm = confusion_matrix(all_cv_targets, all_cv_preds)
    cm_df = pd.DataFrame(cm, index=[f'True_{c}' for c in CLASSES], columns=[f'Pred_{c}' for c in CLASSES])
    cm_df.to_csv('reports/baseline_confusion_matrix.csv')
    print('Saved reports/baseline_confusion_matrix.csv', flush=True)

    # Save detailed prediction logs for error analysis
    pd.DataFrame(cv_records).to_csv('reports/baseline_cv_predictions.csv', index=False)
    print('Saved reports/baseline_cv_predictions.csv', flush=True)

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Baseline MobileNetV3 Confusion Matrix (5-Fold CV)')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45, ha='right')
    plt.yticks(tick_marks, CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('reports/baseline_confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved reports/baseline_confusion_matrix.png', flush=True)

    # Identify top confused pairs
    confused = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                confused.append((CLASSES[i], CLASSES[j], int(cm[i, j])))
    confused.sort(key=lambda x: x[2], reverse=True)
    print('\n--- TOP CONFUSED CLASS PAIRS (True -> Predicted) ---', flush=True)
    for true_c, pred_c, count in confused[:5]:
        total_true = int(np.sum(cm[CLASSES.index(true_c)]))
        pct = (count / total_true) * 100
        print(f'  {true_c} -> {pred_c}: {count} errors ({pct:.1f}% of {true_c})', flush=True)

if __name__ == '__main__':
    main()
