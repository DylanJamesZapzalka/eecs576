print(dataset)

question = "When was Metallica formed?"
question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].detach().numpy()
scores, retrieved_examples = dataset.get_nearest_examples('embeddings', question_embedding, k=10)

doc = retrieved_examples['text'][0]
# answers = ['October 28, 1981']
answers = ['October 2, 1979']


nq_test_df = pd.read_csv('/home/dylanz/eecs_project/datasets/nq-test.qa.csv', sep='\t', header=None)
questions = df[0]
questions = df[0]


# load QA dataset
query_col,answers_col=0,1
queries,answers = [],[]
with open('/home/dylanz/eecs_project/datasets/nq-test.qa.csv') as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        queries.append(normalize_question(row[query_col]))
        answers.append(eval(row[answers_col]))
print(queries)
queries = [queries[idx:idx+32] for idx in range(0,len(queries),32)]
print(queries)
print(answers)