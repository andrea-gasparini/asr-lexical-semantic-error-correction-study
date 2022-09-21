import json
import sys

PRINT = False

with open("../../models/ngrams/sense_ids.txt", "w") as file:

	for jsonl_sample in sys.stdin:

		jsonl_sample = json.loads(jsonl_sample)

		assert "labels" in jsonl_sample, f"Sample does not have a valid 'labels' key"

		sense_ids = jsonl_sample["labels"]
		joined_sense_ids = " ".join([sense_id[0] for sense_id in sense_ids])

		file.write(f"{joined_sense_ids}\n")

		if PRINT:
			print(joined_sense_ids)
		