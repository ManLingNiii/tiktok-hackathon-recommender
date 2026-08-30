"""LLM-driven feature proposal protocol for the History track.

The LLM proposes computations, never identifiers. Specs are validated and a
stable identifier is derived from canonical spec content, so synonymous names
cannot create duplicate candidates.
"""
import hashlib, json, re

BASELINE_FIELDS={'user_id','video_id','author_id','tab','dur_bucket'}
RAW_FIELDS={'user_id','video_id','author_id','tab','date','hourmin','time_ms','duration_ms','long_view','is_click','is_like','is_follow','is_comment','is_forward','is_hate','play_time_ms','profile_stay_time','comment_stay_time','is_profile_enter','is_rand'}
HISTORY_ALLOWED=RAW_FIELDS-BASELINE_FIELDS
FORBIDDEN={'test_label','hidden_test','future_interaction','external_data','pretrained_embedding'}
SUPPORTED_AGGREGATIONS={'smoothed_positive_rate','interaction_count','duration_bucket_affinity','recency_weighted_positive_rate'}
SUPPORTED_WINDOWS={'all_prior','recent'}

SCHEMA={
    'task':'within-user ranking of standard logged impressions',
    'baseline_fields':sorted(BASELINE_FIELDS),
    'available_source_fields':sorted(RAW_FIELDS),
    'history_must_use':'chronological interactions from the official training split only',
    'label':'long_view',
    'metrics':['GAUC','nDCG@5','primary'],
    'forbidden':sorted(FORBIDDEN),
    'requirements':['feature must vary by candidate or interact with candidate fields','no current-row label in its own history','no validation/test labels during feature construction']
}

PROMPT_TEMPLATE='''You are proposing history features for a recommender model.\n\nSchema and constraints:\n{schema}\n\nReturn JSON only with this shape:\n{{"features":[{{"source_columns":[...],"history_scope":"...","time_order":"...","window":{{"type":"...","value":...}},"aggregation":"...","formula":"...","uses_current_row":false,"rationale":"..."}}]}}\n\nDo not return a feature name or id. First specify what to compute. Every source column must be in available_source_fields, history must use only prior training interactions, and the feature must affect candidate ranking.''' 

def proposal_prompt():
    return PROMPT_TEMPLATE.format(schema=json.dumps(SCHEMA,sort_keys=True))

def _canonical(spec):
    return json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=True)

def validate_spec(spec):
    required={'source_columns','history_scope','time_order','window','aggregation','formula','uses_current_row','rationale'}
    missing=required-set(spec)
    if missing: raise ValueError('missing fields: '+','.join(sorted(missing)))
    unknown=set(spec['source_columns'])-RAW_FIELDS
    if unknown: raise ValueError('unknown source columns: '+','.join(sorted(unknown)))
    if set(spec['source_columns']) & FORBIDDEN: raise ValueError('forbidden source field')
    if spec['uses_current_row'] is not False: raise ValueError('uses_current_row must be false')
    if not spec['history_scope'] or not spec['time_order']: raise ValueError('history scope/order required')
    if 'train' not in str(spec['history_scope']).lower(): raise ValueError('history scope must mention train')
    if spec['aggregation'] not in SUPPORTED_AGGREGATIONS: raise ValueError('unsupported aggregation: '+str(spec['aggregation']))
    if not isinstance(spec['window'],dict) or spec['window'].get('type') not in SUPPORTED_WINDOWS: raise ValueError('unsupported window type')
    # A user-only statistic is constant across all candidates for that user,
    # so it cannot change within-user ranking.  Require the computation to
    # reference at least one candidate-side field.
    candidate_fields={'video_id','author_id','tab','duration_ms','dur_bucket'}
    referenced=set(spec['source_columns']) & candidate_fields
    text=' '.join(str(spec[k]).lower() for k in ('history_scope','aggregation','formula'))
    if not referenced and not any(field in text for field in candidate_fields):
        raise ValueError('constraint violation: feature is user-only and cannot affect within-user candidate ranking')
    return True

def canonical_feature_id(spec):
    """Derive ID from computation content, not an LLM-provided name."""
    validate_spec(spec)
    digest=hashlib.sha256(_canonical(spec).encode()).hexdigest()[:16]
    return 'hist_'+digest

def display_name(spec):
    """Human-readable label generated after computation is fixed."""
    cols='_'.join(re.sub(r'[^a-z0-9]+','_',c.lower()).strip('_') for c in sorted(spec['source_columns']))
    agg=re.sub(r'[^a-z0-9]+','_',str(spec['aggregation']).lower()).strip('_')
    return f'{agg}_{cols}'[:120]

def parse_llm_response(text):
    """Parse structured LLM JSON and attach stable IDs after validation."""
    payload=json.loads(text)
    if not isinstance(payload,dict) or not isinstance(payload.get('features'),list): raise ValueError('response must contain features list')
    output=[]; seen=set()
    for raw in payload['features']:
        spec={k:raw[k] for k in ('source_columns','history_scope','time_order','window','aggregation','formula','uses_current_row','rationale')}
        validate_spec(spec); fid=canonical_feature_id(spec)
        if fid in seen: continue
        seen.add(fid); output.append({'feature_id':fid,'display_name':display_name(spec),'spec':spec})
    return output

def propose_features(call_llm):
    """Call an injected LLM function; kept provider-agnostic for reproducibility."""
    response=call_llm(proposal_prompt())
    return parse_llm_response(response)

if __name__=='__main__':
    print(proposal_prompt())
