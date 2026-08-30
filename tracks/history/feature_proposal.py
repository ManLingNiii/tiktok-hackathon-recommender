"""Compatibility definitions for the original six-feature runner.

The autonomous proposal contract lives in ``llm_feature_proposal.py`` and is
used by ``workflow.py``.  This module is retained so the reusable ``train.py``
engine can still be run against the original six-feature comparison.
"""
SCHEMA={'source_fields':['user_id','video_id','author_id','tab','date','duration_ms','long_view'],'history_source':'official train split only','prediction_time_available':True,'forbidden':['test labels','future interactions','cross-user leakage'],'selection_metric':'validation primary'}
DEFAULT_PROPOSALS=['same_author_as_candidate','author_interaction_count','author_interaction_ratio','author_affinity_rate','recent_engagement_rate','dur_bucket_affinity']
def propose_features(): return DEFAULT_PROPOSALS
