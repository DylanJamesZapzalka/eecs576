from fb_dpr_utils import has_answer
from tqdm import tqdm


def get_exact_match_score(question_embeddings, answers_array, dataset, k):

    exact_matches = 0
    for i in tqdm(range(len(question_embeddings)), desc='Evaluating over each question/answer'):
        # Get question and answers
        question_embedding = question_embeddings[i]
        answers = answers_array[i]

        # Get k nearest examples via DPR
        scores, retrieved_examples = dataset.get_nearest_examples('embeddings', question_embedding, k=k)
        retrieved_examples = retrieved_examples['text']

        # Check each of the nearest passages for an exact match
        for retrieved_example in retrieved_examples:
            match = has_answer(answers,retrieved_example)
            if match:
                exact_matches +=1
                break
    
    # Return the score
    score = exact_matches / len(question_embeddings)
    return score