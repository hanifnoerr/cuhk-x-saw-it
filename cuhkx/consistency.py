"""Exact-text combination-to-multi rule."""
import collections
from .core import clip_path, validate_answer

def normalize(text):
    return ' '.join(text.casefold().split())


def apply_rule(questions, predictions):
    result = dict(predictions)
    groups = collections.defaultdict(list)
    for q in questions:
        groups[clip_path(q)].append(q)
    changes = []
    for group in groups.values():
        combinations = [q for q in group if q['category']=='combination' and q['qa_id'] in predictions]
        if len(combinations)!=1:
            continue
        combo = combinations[0]
        letter = predictions[combo['qa_id']]
        validate_answer(combo, letter)
        supported = {normalize(atom) for atom in combo[letter].split(',') if atom.strip()}
        for q in group:
            if q['category']!='multi' or q['qa_id'] not in predictions:
                continue
            previous = predictions[q['qa_id']]
            validate_answer(q, previous)
            additions = {k for k in 'ABCD' if q[k].strip() and normalize(q[k]) in supported}
            candidate = ''.join(sorted(set(previous)|additions))
            validate_answer(q, candidate)
            if candidate!=previous:
                result[q['qa_id']] = candidate
                changes.append({'qa_id':q['qa_id'],'old':previous,'new':candidate,'combination_qa_id':combo['qa_id']})
    return result, changes
