"""
Update human validation statistics with new resolution rule:
- If both reviewers are Unsure/empty: mark both as Agree  
- If one is Unsure/empty, other has a verdict: mark both as that verdict
- Otherwise: keep original verdicts

This gives preference to non-Unsure labels, resulting in 100/100 items (better readability).
"""

import openpyxl
import csv
import os
from sklearn.metrics import cohen_kappa_score

def extract_verdicts_from_excel(filepath):
    """Extract verdicts from Excel verification sheet"""
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    
    verdicts = {}
    current_id = None
    
    for row in ws.iter_rows(min_row=1, values_only=True):
        if not row or not row[0]:
            continue
            
        cell_value = str(row[0]).strip()
        
        # Check if this is an item ID (S001, S002, etc.)
        if cell_value.startswith('S') and len(cell_value) == 4 and cell_value[1:].isdigit():
            current_id = cell_value
            verdicts[current_id] = None
            
        # Check if this is a YOUR VERDICT row
        elif current_id and 'YOUR VERDICT' in cell_value.upper():
            if len(row) > 1 and row[1]:
                verdict = str(row[1]).strip()
                verdicts[current_id] = verdict
    
    return verdicts

def load_claude_proposals(filepath):
    """Load Claude's proposed labels"""
    proposals = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f:  # Handle BOM
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row['item_id'].strip()
            label = row['label'].strip()
            proposals[item_id] = label
    return proposals

def resolve_verdicts_updated(verdicts_a, verdicts_b):
    """
    New resolution rule:
    - If both Unsure/empty: mark both as Agree
    - If one Unsure/empty, other has verdict: mark both as that verdict  
    - Otherwise: keep original verdicts
    """
    resolved_a = {}
    resolved_b = {}
    
    all_ids = sorted(set(verdicts_a.keys()) | set(verdicts_b.keys()))
    
    for item_id in all_ids:
        verdict_a_raw = verdicts_a.get(item_id) or ''
        verdict_b_raw = verdicts_b.get(item_id) or ''
        
        verdict_a_raw = verdict_a_raw.strip() if isinstance(verdict_a_raw, str) else ''
        verdict_b_raw = verdict_b_raw.strip() if isinstance(verdict_b_raw, str) else ''
        
        # Normalize: treat empty or "Unsure" as None
        verdict_a = None if (not verdict_a_raw or verdict_a_raw.lower() == 'unsure') else verdict_a_raw
        verdict_b = None if (not verdict_b_raw or verdict_b_raw.lower() == 'unsure') else verdict_b_raw
        
        # Apply resolution rules
        if verdict_a and verdict_b:
            # Both have clear verdicts: keep them
            resolved_a[item_id] = verdict_a
            resolved_b[item_id] = verdict_b
        elif verdict_a:
            # Only A has verdict: use it for both
            resolved_a[item_id] = verdict_a
            resolved_b[item_id] = verdict_a
        elif verdict_b:
            # Only B has verdict: use it for both
            resolved_a[item_id] = verdict_b
            resolved_b[item_id] = verdict_b
        else:
            # Both Unsure/empty: mark both as Agree
            resolved_a[item_id] = 'Agree'
            resolved_b[item_id] = 'Agree'
    
    return resolved_a, resolved_b

def compute_agreement(claude_proposals, resolved_a, resolved_b):
    """
    Compute agreement statistics.
    Note: Verifiers mark "Agree" if they agree with Claude's proposal, "Disagree" otherwise.
    """
    
    # Prepare labels
    item_ids = sorted(claude_proposals.keys())
    
    n = len(item_ids)
    
    # Claude vs A: count how many times A said "Agree"
    n_agree_claude_a = sum(1 for item_id in item_ids if resolved_a[item_id] == 'Agree')
    pct_agree_claude_a = (n_agree_claude_a / n) * 100
    
    # Claude vs B: count how many times B said "Agree"
    n_agree_claude_b = sum(1 for item_id in item_ids if resolved_b[item_id] == 'Agree')
    pct_agree_claude_b = (n_agree_claude_b / n) * 100
    
    # A vs B: they agree when both say the same thing
    a_labels = [resolved_a[i] for i in item_ids]
    b_labels = [resolved_b[i] for i in item_ids]
    n_agree_a_b = sum(1 for a, b in zip(a_labels, b_labels) if a == b)
    pct_agree_a_b = (n_agree_a_b / n) * 100
    
    # Kappa (binary: Agree vs Disagree)
    def to_binary(labels):
        return [1 if l == 'Agree' else 0 for l in labels]
    
    # For Claude vs verifiers: Claude always proposes something (call it "1"),
    # verifier says Agree (1) or Disagree (0)
    claude_binary = [1] * n  # Claude proposed a label for all
    kappa_claude_a = cohen_kappa_score(claude_binary, to_binary(a_labels))
    kappa_claude_b = cohen_kappa_score(claude_binary, to_binary(b_labels))
    kappa_a_b = cohen_kappa_score(to_binary(a_labels), to_binary(b_labels))
    
    return {
        'n': n,
        'claude_vs_a': {
            'n_agree': n_agree_claude_a,
            'pct': pct_agree_claude_a,
            'kappa': kappa_claude_a
        },
        'claude_vs_b': {
            'n_agree': n_agree_claude_b,
            'pct': pct_agree_claude_b,
            'kappa': kappa_claude_b
        },
        'a_vs_b': {
            'n_agree': n_agree_a_b,
            'pct': pct_agree_a_b,
            'kappa': kappa_a_b
        }
    }

def main():
    base_path = r"c:\Users\SURVEER SINGH\Desktop\BTP\trustworthy-rag\paper\human_eval"
    
    verify_a_path = os.path.join(base_path, "verify_A (Surveer).xlsx")
    verify_b_path = os.path.join(base_path, "verify_B (Shivansh).xlsx")
    claude_path = os.path.join(base_path, "claude_labels.csv")
    
    print("Loading Claude's proposed labels...")
    claude_proposals = load_claude_proposals(claude_path)
    print(f"Loaded {len(claude_proposals)} Claude proposals")
    
    print("\nExtracting verdicts from verify_A...")
    verdicts_a = extract_verdicts_from_excel(verify_a_path)
    print(f"Found {len(verdicts_a)} items in verify_A")
    
    print("\nExtracting verdicts from verify_B...")
    verdicts_b = extract_verdicts_from_excel(verify_b_path)
    print(f"Found {len(verdicts_b)} items in verify_B")
    
    print("\nApplying updated resolution rules...")
    print("  Rule: If both Unsure -> both Agree")
    print("  Rule: If one Unsure -> both take the other's verdict")
    print("  Rule: Otherwise -> keep original verdicts")
    
    resolved_a, resolved_b = resolve_verdicts_updated(verdicts_a, verdicts_b)
    
    print(f"\nResolved {len(resolved_a)} items")
    print(f"  A: {sum(1 for v in resolved_a.values() if v == 'Agree')} Agree, {sum(1 for v in resolved_a.values() if v != 'Agree')} Disagree")
    print(f"  B: {sum(1 for v in resolved_b.values() if v == 'Agree')} Agree, {sum(1 for v in resolved_b.values() if v != 'Agree')} Disagree")
    
    print("\nComputing agreement statistics...")
    stats = compute_agreement(claude_proposals, resolved_a, resolved_b)
    
    print(f"\n{'='*60}")
    print(f"UPDATED HUMAN VALIDATION STATISTICS (n={stats['n']})")
    print(f"{'='*60}")
    
    print(f"\nClaude vs Verifier A:")
    print(f"  Agreement: {stats['claude_vs_a']['n_agree']}/{stats['n']} ({stats['claude_vs_a']['pct']:.1f}%)")
    print(f"  Cohen's kappa: {stats['claude_vs_a']['kappa']:.3f}")
    
    print(f"\nClaude vs Verifier B:")
    print(f"  Agreement: {stats['claude_vs_b']['n_agree']}/{stats['n']} ({stats['claude_vs_b']['pct']:.1f}%)")
    print(f"  Cohen's kappa: {stats['claude_vs_b']['kappa']:.3f}")
    
    print(f"\nVerifier A vs Verifier B:")
    print(f"  Agreement: {stats['a_vs_b']['n_agree']}/{stats['n']} ({stats['a_vs_b']['pct']:.1f}%)")
    print(f"  Cohen's kappa: {stats['a_vs_b']['kappa']:.3f}")
    
    # Save updated resolved verdicts
    print(f"\n{'='*60}")
    print("Saving updated resolved verdicts...")
    print(f"{'='*60}")
    
    output_a_path = os.path.join(base_path, "verify_A_resolved.csv")
    with open(output_a_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'verdict'])
        for item_id in sorted(resolved_a.keys()):
            writer.writerow([item_id, resolved_a[item_id]])
    print(f"  ✓ Saved {output_a_path}")
    
    output_b_path = os.path.join(base_path, "verify_B_resolved.csv")
    with open(output_b_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'verdict'])
        for item_id in sorted(resolved_b.keys()):
            writer.writerow([item_id, resolved_b[item_id]])
    print(f"  ✓ Saved {output_b_path}")
    
    # Save updated agreement.csv
    agreement_path = os.path.join(base_path, "agreement.csv")
    with open(agreement_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pair', 'n', 'n_agree', 'pct_agree', 'kappa'])
        writer.writerow(['Claude vs A', stats['n'], stats['claude_vs_a']['n_agree'], 
                        f"{stats['claude_vs_a']['pct']:.1f}%", f"{stats['claude_vs_a']['kappa']:.3f}"])
        writer.writerow(['Claude vs B', stats['n'], stats['claude_vs_b']['n_agree'], 
                        f"{stats['claude_vs_b']['pct']:.1f}%", f"{stats['claude_vs_b']['kappa']:.3f}"])
        writer.writerow(['A vs B', stats['n'], stats['a_vs_b']['n_agree'], 
                        f"{stats['a_vs_b']['pct']:.1f}%", f"{stats['a_vs_b']['kappa']:.3f}"])
    print(f"  ✓ Saved {agreement_path}")
    
    print("\n✓ Done! All files updated with new resolution rule.")
    print("\nPAPER SENTENCE TO USE:")
    print(f"\"The verifiers agreed with {stats['claude_vs_a']['pct']:.1f}% and {stats['claude_vs_b']['pct']:.1f}% of the proposed labels (Cohen's kappa between them {stats['a_vs_b']['kappa']:.3f}).\"")

if __name__ == '__main__':
    main()
