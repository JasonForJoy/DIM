
test_out_filename = "persona_test_out.txt"

with open(test_out_filename, 'r') as f:
	cur_q_id = None
	num_query = 0
	recall = {"recall@1": 0,
			  "recall@2": 0,
			  "recall@5": 0,
			  "recall@10": 0}

	lines = f.readlines()
	for line in lines[1:]:
		line = line.strip().split('\t')
		line = [float(ele) for ele in line]

		if cur_q_id is None:
			cur_q_id = line[0]
			num_query += 1
		elif line[0] != cur_q_id:
			cur_q_id = line[0]
			num_query += 1

		if line[4] == 1.0:
			rank = line[3]

			if rank <= 1:
				recall["recall@1"] += 1
			if rank <= 2:
				recall["recall@2"] += 1
			if rank <= 5:
				recall["recall@5"] += 1
			if rank <= 10:
				recall["recall@10"] += 1

	recall["recall@1"] = recall["recall@1"] / float(num_query)
	recall["recall@2"] = recall["recall@2"] / float(num_query)
	recall["recall@5"] = recall["recall@5"] / float(num_query)
	recall["recall@10"] = recall["recall@10"] / float(num_query)
	print("num_query = {}".format(num_query))
	print("recall@1 = {}".format(recall["recall@1"]))
	print("recall@2 = {}".format(recall["recall@2"]))
	print("recall@5 = {}".format(recall["recall@5"]))
	print("recall@10 = {}".format(recall["recall@10"]))
