import json

print("Loading questions...")
with open('./val/v2_OpenEnded_mscoco_val2014_questions.json', encoding='utf-8') as json_data:
	val_qs = json.load(json_data)

questions = val_qs['questions']

file = []
for q in questions:
	q_id = q['question_id']
	entry = {"answer": "yes", "question_id": q_id}
	file.append(entry)

with open('v2_OpenEnded_mscoco_val2014_fake_results.json', 'w') as f:
	json.dump(file, f)