import os
import random
import numpy as np
from nltk.tokenize import WordPunctTokenizer


def tokenize(text):
	return WordPunctTokenizer().tokenize(text)


def data_process_none(input_path, output_path, fname):
	
	dialogues = []
	dialogue = []
	with open(os.path.join(input_path, fname), "r") as f:
		for line in f:
			line = line.decode('utf-8').strip()
			if line.split()[0] == "1":  # new dialogue
				dialogues.append(dialogue)
				dialogue = []
			dialogue.append(line)

		dialogues.append(dialogue)
		dialogues.remove([])
	print("{} is composed of {} dialogues".format(fname, len(dialogues)))

	context_candidates = []
	for dialogue in dialogues:
		context_history = []
		for turn in dialogue:
			fields = turn.split("\t")
			context = " ".join(tokenize(fields[0])[1:])
			response = fields[1]
			candidates = fields[-1].split("|")
			random.shuffle(candidates)
			label = candidates.index(response)

			context_history.append(context)
			# (context, candidates, label, partner's persona, your persona)
			context_candidates.append( [" _eos_ ".join(context_history) + " _eos_", 
									    "|".join(candidates), 
									    str(label),
									    "NA", 
									    "NA"] )
			context_history.append(response)

	print("{} is composed of {} context-candidates".format(fname, len(context_candidates)))

	with open(os.path.join(output_path, "processed_{}".format(fname)), "w") as f:
		print("Saving dataset to processed_{} ...".format(fname))
		for dialogue in context_candidates:
			f.write(("\t".join(dialogue) + "\n").encode('utf-8'))


def data_process_self(input_path, output_path, fname):

	dialogues = []
	dialogue = []
	with open(os.path.join(input_path, fname), "r") as f:
		for line in f:
			line = line.decode('utf-8').strip()
			if line.split()[0] == "1":  # new dialogue
				dialogues.append(dialogue)
				dialogue = []
			dialogue.append(line)

		dialogues.append(dialogue)
		dialogues.remove([])
	print("{} is composed of {} dialogues".format(fname, len(dialogues)))

	context_candidates = []
	for dialogue in dialogues:
		persona = []
		context_history = []
		for line in dialogue:
			fields = line.strip().split("\t")

			if len(fields) == 1:
				persona.append((" ").join(tokenize(fields[0])[4:]))
			if len(fields) == 4:
				context = " ".join(tokenize(fields[0])[1:])
				response = fields[1]
				candidates = fields[-1].split("|")
				random.shuffle(candidates)
				label = candidates.index(response)

				context_history.append(context)
				# (context, candidates, label, partner's persona, your persona)
	 			context_candidates.append( [" _eos_ ".join(context_history) + " _eos_", 
		 									"|".join(candidates), 
		 									str(label),
		 									"NA", 
		 									"|".join(persona)] )
	 			context_history.append(response)
	print("{} is composed of {} context-candidates".format(fname, len(context_candidates)))

	with open(os.path.join(output_path, "processed_{}".format(fname)), "w") as f:
		print("Saving dataset to processed_{} ...".format(fname))
		for dialogue in context_candidates:
			f.write(("\t".join(dialogue) + "\n").encode('utf-8'))


def data_process_other(input_path, output_path, fname):

	dialogues = []
	dialogue = []
	with open(os.path.join(input_path, fname), "r") as f:
		for line in f:
			line = line.decode('utf-8').strip()
			if line.split()[0] == "1":  # new dialogue
				dialogues.append(dialogue)
				dialogue = []
			dialogue.append(line)

		dialogues.append(dialogue)
		dialogues.remove([])
	print("{} is composed of {} dialogues".format(fname, len(dialogues)))

	context_candidates = []
	for dialogue in dialogues:
		persona = []
		context_history = []
		for line in dialogue:
			fields = line.strip().split("\t")

			if len(fields) == 1:
				persona.append((" ").join(tokenize(fields[0])[6:]))
			if len(fields) == 4:
				context = " ".join(tokenize(fields[0])[1:])
				response = fields[1]
				candidates = fields[-1].split("|")
				random.shuffle(candidates)
				label = candidates.index(response)

				context_history.append(context)
				# (context, candidates, label, partner's persona, your persona)
	 			context_candidates.append( [" _eos_ ".join(context_history) + " _eos_", 
		 									"|".join(candidates), 
		 									str(label),
		 									"|".join(persona),
		 									"NA"] )
	 			context_history.append(response)
	print("{} is composed of {} context-candidates".format(fname, len(context_candidates)))

	with open(os.path.join(output_path, "processed_{}".format(fname)), "w") as f:
		print("Saving dataset to processed_{} ...".format(fname))
		for dialogue in context_candidates:
			f.write(("\t".join(dialogue) + "\n").encode('utf-8'))


def data_process_both(input_path, output_path, fname):

	dialogues = []
	dialogue = []
	with open(os.path.join(input_path, fname), "r") as f:
		for line in f:
			line = line.decode('utf-8').strip()
			if line.split()[0] == "1":  # new dialogue
				dialogues.append(dialogue)
				dialogue = []
			dialogue.append(line)

		dialogues.append(dialogue)
		dialogues.remove([])
	print("{} is composed of {} dialogues".format(fname, len(dialogues)))

	context_candidates = []
	for dialogue in dialogues:
		self_persons = []
		other_persona = []
		context_history = []
		for line in dialogue:
			fields = line.strip().split("\t")

			if len(fields) == 1:
				if fields[0].split()[1] == "your":
					self_persons.append((" ").join(tokenize(fields[0])[4:]))
				if fields[0].split()[1] == "partner's":
					other_persona.append((" ").join(tokenize(fields[0])[6:]))
			if len(fields) == 4:
				context = " ".join(tokenize(fields[0])[1:])
				response = fields[1]
				candidates = fields[-1].split("|")
				random.shuffle(candidates)
				label = candidates.index(response)

				context_history.append(context)
				# (context, candidates, label, partner's persona, your persona)
	 			context_candidates.append( [" _eos_ ".join(context_history) + " _eos_", 
		 									"|".join(candidates), 
		 									str(label),
		 									"|".join(other_persona),
		 									"|".join(self_persons)] )
	 			context_history.append(response)
	print("{} is composed of {} context-candidates".format(fname, len(context_candidates)))

	with open(os.path.join(output_path, "processed_{}".format(fname)), "w") as f:
		print("Saving dataset to processed_{} ...".format(fname))
		for dialogue in context_candidates:
			f.write(("\t".join(dialogue) + "\n").encode('utf-8'))

if __name__ == '__main__':

	input_path = "./personachat"
	output_path = "./personachat_processed"

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	files = [file for file in os.listdir(input_path)]
	files_none = [file for file in files if file.split("_")[1] == "none"]
	files_self = [file for file in files if file.split("_")[1] == "self"]
	files_other = [file for file in files if file.split("_")[1] == "other"]
	files_both = [file for file in files if file.split("_")[1] == "both"]

	print("There are {} files to process.\nStart processing data ...".format(len(files)))

	for file in files_none:
		print("Preprocessing {} ...".format(file))
		data_process_none(input_path, output_path, file)
		print("="*60)
	
	for file in files_self:
		print("Preprocessing {} ...".format(file))
		data_process_self(input_path, output_path, file)
		print("="*60)
	
	for file in files_other:
		print("Preprocessing {} ...".format(file))
		data_process_other(input_path, output_path, file)
		print("="*60)
	
	for file in files_both:
		print("Preprocessing {} ...".format(file))
		data_process_both(input_path, output_path, file)
		print("="*60)
	
	print("data preprocess done!")
