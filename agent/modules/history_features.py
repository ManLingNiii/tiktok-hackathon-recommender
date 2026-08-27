"""Leakage-safe user history features.

Rows must be chronological. A row's features are computed before its label is
added to the history, so validation rows never update training history.
"""
from collections import defaultdict


def add_history_features(rows):
    """Return rows with user/category/author history counts added.

    Expected row format: (date, user_id, video_id, author_id, tab,
    duration_ms, long_view). The returned rows append feature values before
    the current row is observed.
    """
    users = defaultdict(lambda: [0, 0])
    user_category = defaultdict(lambda: [0, 0])
    user_author = defaultdict(lambda: [0, 0])
    output = []
    for row in sorted(rows, key=lambda x: x[0]):
        date, user, video, author, tab, duration, label = row
        uc = user_category[(user, tab)]
        ua = user_author[(user, author)]
        seen, positives = users[user]
        output.append(tuple(row) + (seen, positives, uc[0], uc[1], ua[0], ua[1]))
        users[user] = [seen + 1, positives + int(label)]
        uc[0] += 1; uc[1] += int(label)
        ua[0] += 1; ua[1] += int(label)
    return output
