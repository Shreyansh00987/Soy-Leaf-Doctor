import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def main():
    preds_file = 'reports/baseline_cv_predictions.csv'
    if not os.path.exists(preds_file):
        print(f"File {preds_file} not found!")
        return

    df_preds = pd.read_csv(preds_file)
    img_dir = 'data/p4-soy-leaf-health/images'
    os.makedirs('reports', exist_ok=True)

    # 1. Filter errors
    errors = df_preds[df_preds['is_error'] == 1].copy()
    print(f"Total validation samples: {len(df_preds)}")
    print(f"Total errors: {len(errors)} (Error rate: {len(errors)/len(df_preds)*100:.2f}%)")

    # Sort by loss descending (highest cross-entropy loss)
    highest_loss = errors.sort_values(by='loss', ascending=False)
    
    # Sort by pred_prob descending (most confidently wrong)
    confidently_wrong = errors.sort_values(by='pred_prob', ascending=False)

    print("\n--- Top 10 Most Confidently Wrong Predictions ---")
    top_confident = confidently_wrong.head(10)
    for idx, row in top_confident.iterrows():
        print(f"ID: {row['image_id']} | True: {row['true_label']} ({row['true_prob']:.3f}) | Pred: {row['pred_label']} ({row['pred_prob']:.3f}) | Loss: {row['loss']:.2f}")

    # 2. Flag likely mislabeled or ambiguous samples
    # Criteria:
    # 1) Exact duplicate contradictions (from our hash audit)
    # 2) Model predicted another class with probability > 0.85 and true class < 0.05
    # 3) Samples with loss > 3.0
    
    # Exact duplicate pairs from Step 1
    dup_pairs = [
        ('soy_01056', 'soy_02254'), ('soy_00188', 'soy_00871'),
        ('soy_00235', 'soy_02147'), ('soy_00024', 'soy_00520'),
        ('soy_00459', 'soy_02057'), ('soy_01495', 'soy_00622'),
        ('soy_01432', 'soy_00366'), ('soy_01533', 'soy_02236'),
        ('soy_01006', 'soy_00629'), ('soy_00127', 'soy_01254'),
        ('soy_00052', 'soy_00212'), ('soy_02129', 'soy_00990'),
        ('soy_00098', 'soy_00784')
    ]
    exact_dup_ids = set()
    for u, v in dup_pairs:
        exact_dup_ids.add(u)
        exact_dup_ids.add(v)

    flagged_candidates = []
    for idx, row in df_preds.iterrows():
        img_id = row['image_id']
        reason = []
        confidence_flag = "NORMAL"

        if img_id in exact_dup_ids:
            reason.append("EXACT_DUPLICATE_CONTRADICTORY_LABEL")
            confidence_flag = "HIGH_MISLABEL_SUSPECT"

        if row['is_error'] == 1 and row['pred_prob'] >= 0.85 and row['true_prob'] < 0.05:
            reason.append(f"EXTREME_CONFIDENCE_WRONG (P_pred={row['pred_prob']:.3f}, P_true={row['true_prob']:.3f})")
            confidence_flag = "HIGH_MISLABEL_SUSPECT"
        elif row['is_error'] == 1 and row['loss'] >= 2.5:
            reason.append(f"HIGH_LOSS ({row['loss']:.2f})")
            if confidence_flag == "NORMAL":
                confidence_flag = "AMBIGUOUS_SAMPLE"

        if reason:
            flagged_candidates.append({
                'image_id': img_id,
                'fold': row['fold'],
                'true_label': row['true_label'],
                'predicted_label': row['pred_label'],
                'true_prob': round(row['true_prob'], 4),
                'predicted_prob': round(row['pred_prob'], 4),
                'loss': round(row['loss'], 4),
                'flag': confidence_flag,
                'reasons': " | ".join(reason)
            })

    flagged_df = pd.DataFrame(flagged_candidates)
    flagged_csv = 'reports/mislabeled_candidates.csv'
    flagged_df.to_csv(flagged_csv, index=False)
    print(f"\nSaved {len(flagged_df)} flagged samples to {flagged_csv} for manual review.")
    print("Flag summary:\n", flagged_df['flag'].value_counts().to_dict())

    # 3. Create a Visual Grid of Top 16 Most-Confidently Wrong Images
    top_16 = confidently_wrong.head(16).reset_index(drop=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(16):
        ax = axes[i]
        if i < len(top_16):
            row = top_16.iloc[i]
            img_p = os.path.join(img_dir, f"{row['image_id']}.jpg")
            with Image.open(img_p) as img:
                ax.imshow(img)
            title = f"ID: {row['image_id']}\nTrue: {row['true_label']} ({row['true_prob']:.2f})\nPred: {row['pred_label']} ({row['pred_prob']:.2f})"
            ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        ax.axis('off')

    plt.suptitle("Top 16 Most-Confidently-Wrong Validation Samples (Baseline Model)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    grid_path = 'reports/top_errors_grid.png'
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Saved visual error grid to {grid_path}")

if __name__ == '__main__':
    main()
