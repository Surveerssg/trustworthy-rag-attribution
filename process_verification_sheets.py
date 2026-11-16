import openpyxl
import csv
import os

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

def resolve_verdicts(verdicts_a, verdicts_b):
    """Resolve Unsure and empty verdicts according to rules:
    - If Unsure in one sheet, take label from other sheet
    - If empty, default to 'Agree'
    """
    resolved = {}
    all_ids = set(verdicts_a.keys()) | set(verdicts_b.keys())
    
    for item_id in sorted(all_ids):
        verdict_a = verdicts_a.get(item_id)
        verdict_b = verdicts_b.get(item_id)
        
        # Convert to string and strip, handle None
        verdict_a = verdict_a.strip() if verdict_a else ''
        verdict_b = verdict_b.strip() if verdict_b else ''
        
        # Normalize verdicts
        if verdict_a.lower() == 'unsure' or not verdict_a:
            verdict_a = None
        if verdict_b.lower() == 'unsure' or not verdict_b:
            verdict_b = None
            
        # Resolution logic
        if verdict_a and verdict_b:
            # Both have clear verdicts, use A's (or we could check agreement)
            resolved[item_id] = {'A': verdict_a, 'B': verdict_b}
        elif verdict_a:
            # Only A has clear verdict
            resolved[item_id] = {'A': verdict_a, 'B': verdict_a}
        elif verdict_b:
            # Only B has clear verdict
            resolved[item_id] = {'A': verdict_b, 'B': verdict_b}
        else:
            # Both Unsure or empty, default to Agree
            resolved[item_id] = {'A': 'Agree', 'B': 'Agree'}
    
    return resolved

def main():
    base_path = r"c:\Users\SURVEER SINGH\Desktop\BTP\trustworthy-rag\paper\human_eval"
    
    verify_a_path = os.path.join(base_path, "verify_A (Surveer).xlsx")
    verify_b_path = os.path.join(base_path, "verify_B (Shivansh).xlsx")
    
    print("Extracting verdicts from verify_A...")
    verdicts_a = extract_verdicts_from_excel(verify_a_path)
    print(f"Found {len(verdicts_a)} items in verify_A")
    
    print("\nExtracting verdicts from verify_B...")
    verdicts_b = extract_verdicts_from_excel(verify_b_path)
    print(f"Found {len(verdicts_b)} items in verify_B")
    
    print("\nResolving Unsure and empty verdicts...")
    resolved = resolve_verdicts(verdicts_a, verdicts_b)
    
    # Count statistics
    agree_a = sum(1 for v in resolved.values() if v['A'].lower() == 'agree')
    disagree_a = sum(1 for v in resolved.values() if v['A'].lower() == 'disagree')
    agree_b = sum(1 for v in resolved.values() if v['B'].lower() == 'agree')
    disagree_b = sum(1 for v in resolved.values() if v['B'].lower() == 'disagree')
    
    # Check agreement between A and B
    agreement = sum(1 for v in resolved.values() if v['A'].lower() == v['B'].lower())
    
    print(f"\nResolved {len(resolved)} items")
    print(f"\nAnnotator A: {agree_a} Agree, {disagree_a} Disagree")
    print(f"Annotator B: {agree_b} Agree, {disagree_b} Disagree")
    print(f"Inter-annotator agreement: {agreement}/{len(resolved)} ({100*agreement/len(resolved):.1f}%)")
    
    # Save resolved verdicts to CSV files
    output_a = os.path.join(base_path, "verify_A_resolved.csv")
    output_b = os.path.join(base_path, "verify_B_resolved.csv")
    
    with open(output_a, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'verdict'])
        for item_id in sorted(resolved.keys()):
            writer.writerow([item_id, resolved[item_id]['A']])
    
    with open(output_b, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'verdict'])
        for item_id in sorted(resolved.keys()):
            writer.writerow([item_id, resolved[item_id]['B']])
    
    print(f"\nSaved resolved verdicts to:")
    print(f"  {output_a}")
    print(f"  {output_b}")
    
    # Print sample of resolved verdicts
    print("\nSample of resolved verdicts (first 10):")
    for item_id in sorted(resolved.keys())[:10]:
        print(f"  {item_id}: A={resolved[item_id]['A']}, B={resolved[item_id]['B']}")

if __name__ == "__main__":
    main()
