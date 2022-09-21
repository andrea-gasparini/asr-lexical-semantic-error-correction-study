import argparse
from tqdm import tqdm

"""
There is a small problem that 🤗 Transformers will not be happy about later on.
The n-gram correctly includes a "Unknown" or <unk>, as well as a begin-of-sentence, <s> token, but no end-of-sentence, </s> token.
This sadly has to be corrected currently after the build.

We can simply add the end-of-sentence token by adding the line 0 </s> $begin_of_sentence_score below the <s> token
and increasing the ngram 1 count by 1.
"""

def main(data_path: str, ngram_size: int) -> None:

	file_prefix = f"{data_path}{ngram_size}gram"

	with open(f"{file_prefix}.arpa", "r") as read_file, open(f"{file_prefix}_hf.arpa", "w") as write_file:

		has_added_eos = False
		num_lines = sum(1 for _ in open(f"{file_prefix}.arpa", "r"))

		for line in tqdm(read_file, total=num_lines):

			if not has_added_eos and "ngram 1=" in line:
				count=line.strip().split("=")[-1]
				write_file.write(line.replace(f"{count}", f"{int(count)+1}"))

			elif not has_added_eos and "<s>" in line:
				write_file.write(line)
				write_file.write(line.replace("<s>", "</s>"))
				has_added_eos = True

			else:
				write_file.write(line)

if __name__ == "__main__":	

	parser = argparse.ArgumentParser("KenLM adapter for Transformers 🤗")
	# required
	parser.add_argument("--data-path", help="The path to the data directory.", type=str, required=True)
	# default + not required
	parser.add_argument("--ngram-size", help="The size of the ngram to take into consideration.", type=int, default=4)

	args = parser.parse_args()

	main(data_path=args.data_path, ngram_size=args.ngram_size)