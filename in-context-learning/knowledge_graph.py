import pandas as pd
from functools import lru_cache

# Load data
SUBRELOBJ = pd.read_csv('SUBRELOBJ.csv')

# Create sets for faster lookups
unique_entities = set(pd.concat([SUBRELOBJ['Subject'], SUBRELOBJ['Object']]).unique())
print("Number of unique entities : {}".format(len(unique_entities)))

@lru_cache(maxsize=1000)
def get_relevant_entities(query):
    return {ent for ent in unique_entities if query.lower() in ent.lower()}

def query_df(entities, rel, k):
    entities = set(entities)
    mask = (SUBRELOBJ['Subject'].isin(entities)) & (SUBRELOBJ['Rel'] == rel)
    filtered_df = SUBRELOBJ[mask]
    return filtered_df.groupby('Object')['Count'].agg('sum').nlargest(k)

def find_overlaps(apl_query):
    relevant_apl_entities = get_relevant_entities(apl_query)
    
    if not relevant_apl_entities:
        print("No relevant APL query results")
        return None
    
    # Get top 5 MAT and PRO for the APL
    top_apl_mat = query_df(relevant_apl_entities, 'APL-CHM', 5)
    
    # Exclude general properties for PRO
    excluded_properties = {'Activity', 'Structure', 'Efficiency', 'Capacity', 'Stability', 'Density','Structural', 'Compositions', 'Morphology', 'Molecular Structure'}
    pro_mask = (SUBRELOBJ['Subject'].isin(relevant_apl_entities)) & \
               (SUBRELOBJ['Rel'] == 'APL-PRO') & \
               (~SUBRELOBJ['Object'].isin(excluded_properties))
    
    filtered_pro_df = SUBRELOBJ[pro_mask]
    top_apl_pro = filtered_pro_df.groupby('Object')['Count'].agg('sum').nlargest(5)
    
    # Find best MAT-PRO combination
    best_combo = None
    highest_count = 0
    
    # Try all combinations of MAT and PRO
    for mat in top_apl_mat.index:
        for pro in top_apl_pro.index:
            mat_pro_mask = (SUBRELOBJ['Subject'] == mat) & \
                          (SUBRELOBJ['Object'] == pro) & \
                          (SUBRELOBJ['Rel'] == 'CHM-PRO')
            
            if SUBRELOBJ[mat_pro_mask].empty:
                continue
                
            current_count = SUBRELOBJ[mat_pro_mask]['Count'].iloc[0]
            
            if current_count > highest_count:
                highest_count = current_count
                best_combo = (mat, pro)
    
    if not best_combo:
        print("No valid MAT-PRO combinations found")
        return None
        
    best_mat, best_pro = best_combo
    
    # Get top 5 CMT for both MAT and PRO
    mat_cmt = query_df({best_mat}, 'CHM-CMT', 5)
    pro_cmt = query_df({best_pro}, 'PRO-CMT', 5)
    
    # Find overlapping CMTs and their combined counts
    common_cmts = set(mat_cmt.index) & set(pro_cmt.index)
    
    if common_cmts:
        # For overlapping CMTs, sum their counts from both relationships
        cmt_counts = {}
        for cmt in common_cmts:
            cmt_counts[cmt] = mat_cmt[cmt] + pro_cmt[cmt]
        best_cmt = max(cmt_counts.items(), key=lambda x: x[1])[0]
    else:
        # If no overlap, use top MAT-CMT
        best_cmt = mat_cmt.index[0]
    
    # Repeat the same process for SMT
    mat_smt = query_df({best_mat}, 'CHM-SMT', 5)
    pro_smt = query_df({best_pro}, 'PRO-SMT', 5)
    
    common_smts = set(mat_smt.index) & set(pro_smt.index)
    
    if common_smts:
        smt_counts = {}
        for smt in common_smts:
            smt_counts[smt] = mat_smt[smt] + pro_smt[smt]
        best_smt = max(smt_counts.items(), key=lambda x: x[1])[0]
    else:
        # If no overlap, use top MAT-SMT
        best_smt = mat_smt.index[0] if not mat_smt.empty else "No SMT found"

    result = {
        'APL': apl_query,
        'CHM': best_mat,
        'PRO': best_pro,
        'CMT': best_cmt,
        'SMT': best_smt
    }

    print(f"APL: {result['APL']}, CHM: {result['CHM']}, PRO: {result['PRO']}, CMT: {result['CMT']}, SMT: {result['SMT']}")
    
    return [result]

results = find_overlaps('oled')