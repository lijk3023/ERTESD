import json
import argparse
import openai
openai.api_key = 'sk-'
from transformers import (BertTokenizerFast)
import time
import re
import sys

def predict(prompt):
    retry_count = 100
    retry_interval = 1
    for _ in range(retry_count):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "NLP professor especially in specific domain"},
                          {"role": "user", "content": prompt}],
                temperature=0.7
            )
            msg = response.choices[0].message["content"].strip()
            return msg
        except openai.error.RateLimitError as e:
            print("Exceeded OpenAI API call rate.", e)
            print('Please retry...')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)
        except Exception as e:
            print("Task execution error:", e)
            print('Please retry...')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)

def detect_answer(s):
    matches = re.findall(r"\(([A-Z])\)", s)
    return matches if matches else None

def find_incorrect_triples_ignore_space_with_sets2(data, tokenizer):
    incorrect_triples = []
    all_positives = 0
    for entry in data:
        text = entry['text']
        gold_spo_set = [(list((tokenizer(subject)['input_ids'])),predicate,list(tokenizer(obj)['input_ids']))
                        for subject, predicate, obj in entry['gold_spo_list']]
        pred_spo_set = [(list((tokenizer(subject)['input_ids'])),predicate,list(tokenizer(obj)['input_ids']))
                        for subject, predicate, obj in entry['pred_spo_list']]
        all_positives = all_positives + len(pred_spo_set)
        false_positives = []
        for pred_spo in pred_spo_set:
            flag = False
            for gold_spo in gold_spo_set:
                if pred_spo[0] == gold_spo[0] and pred_spo[2] == gold_spo[2] and pred_spo[1] == gold_spo[1]:
                    flag = True
                    break
            if not flag:
                false_positives.append(pred_spo)
        false_negatives = []
        for gold_spo in gold_spo_set:
            flag = False
            for pred_spo in pred_spo_set:
                if pred_spo[0] == gold_spo[0] and pred_spo[2] == gold_spo[2] and pred_spo[1] == gold_spo[1]:
                    flag = True
                    break
            if not flag:
                false_negatives.append(gold_spo)

        if false_positives or false_negatives:
            incorrect_entry = {
                'text': text,
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives),
                'gold_spo_set': list(gold_spo_set),
                'pred_spo_set': list(pred_spo_set)
            }
            incorrect_triples.append(incorrect_entry)
        else:
            incorrect_entry = {
                'text': text,
                'false_positives': [],
                'false_negatives': [],
                'gold_spo_set': list(gold_spo_set),
                'pred_spo_set': list(pred_spo_set)
            }

    return incorrect_triples, all_positives

def count_true_and_false_positives_and_negatives2(incorrect_triples):
    total_false_positives = 0
    total_false_negatives = 0

    for entry in incorrect_triples:
        total_false_positives += len(entry['false_positives'])
        total_false_negatives += len(entry['false_negatives'])

    return total_false_positives, total_false_negatives

def calculate_f1(data, tokenizer):

    incorrect_triples_ignore_space_with_sets, all_positives = find_incorrect_triples_ignore_space_with_sets2(data, tokenizer)

    total_false_positives, total_false_negatives = count_true_and_false_positives_and_negatives2(
        incorrect_triples_ignore_space_with_sets)

    precision = (all_positives - total_false_positives) / all_positives
    recall = (all_positives - total_false_positives) / (all_positives - total_false_positives + total_false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    precision = round(precision*100, 2)
    recall = round(recall*100, 2)
    f1 = round(f1*100, 2)
    print("\rdone calculating!          ", end='')
    sys.stdout.flush()
    print("\n")
    print("all_positives:", all_positives, "\tfalse_positives:", total_false_positives, "\tfalse_negatives:", total_false_negatives)
    print("precision: ", precision, "%")
    print("recall: ", recall, "%")
    print("f1: ", f1, "%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='1.json', help='input file path')
    parser.add_argument('--goldpath', type=str, default='2.json', help='gold file path')
    parser.add_argument('--dataset', type=str, default='ccl', help='dataset')
    args = parser.parse_args()

    if args.dataset == 'ccl':
        sub_prompt = "Given a text from the domain of automotive field and sub-domain of automotive fault data: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} must be a part or attribute entity and is the true subject of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        obj_prompt = "Given a text from the domain of automotive field and sub-domain of automotive fault data: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} must be a kind of failure, and is the true object of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        rel_prompt = "Given a text from the domain of automotive field and sub-domain of automotive fault data: {} .\
    When performing an NLP task for text analysis, please make a simple judgment on the subject: {}, and the object: {} in the following sentences. \
    Please select the correct answer: \
    A. Belongs to relation part failure.  \
    B. Belongs to relation attribute failure. \
    C. None of the above. "
        rel_dict = {'A': 'part_failure', 'B': 'attr_failure', 'C': 'None'}

    elif args.dataset == 'scierc':
        sub_prompt = "Given a text from the domain of scientific research field and sub-domain of SCI research paper abstract: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} is the true subject of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        obj_prompt = "Given a text from the domain of scientific research field and sub-domain of SCI research paper abstract: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} is the true object of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        rel_prompt = "Given a text from the domain of scientific research field and sub-domain of SCI research paper abstract: {} .\
    When performing an NLP task for text analysis, please make a simple judgment on the subject: {}, and the object: {} in the following sentences. \
    Please select the correct answer: \
    A. Belongs to relation COMPARE. \
    B. Belongs to relation CONJUNCTION. \
    C. Belongs to relation EVALUATE-FOR. \
    D. Belongs to relation FEATURE-OF. \
    E. Belongs to relation HYPONYM-OF. \
    F. Belongs to relation PART-OF. \
    G. Belongs to relation USED-FOR. \
    H. None of the above. "
        rel_dict = {'A': 'COMPARE', 'B': 'CONJUNCTION', 'C': 'EVALUATE-FOR', 'D': 'FEATURE-OF', 'E': 'HYPONYM-OF', 'F': 'PART-OF', 'G': 'USED-FOR', 'H': 'None'}

    elif args.dataset == 'conll':
        sub_prompt = "Given a text from the domain of news industry and sub-domain of daily newspaper: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} is the true subject of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        obj_prompt = "Given a text from the domain of news industry and sub-domain of daily newspaper: {} .\
    When performing an NLP task for text analysis, consider the subject: {} and object: {} in the sentence, as well as the potential triple sentence they form. \
    For the following statement, please make a judgment: {} is the true object of the triple and the boundaries of the entity are neither missing nor redundant. The relationship {}{} belongs to relation {}. Please select: \
    A. The statement above is correct. \
    B. There are incorrect elements in the statement above. "
        rel_prompt = "Given a text from the domain of news industry and sub-domain of daily newspaper: {} .\
    When performing an NLP task for text analysis, please make a simple judgment on the subject: {}, and the object: {} in the following sentences. \
    Please select the correct answer: \
    A. Belongs to relation Work for. \
    B. Belongs to relation Kill. \
    C. Belongs to relation Organization based in. \
    D. Belongs to relation Live in. \
    E. Belongs to relation Located in. \
    F. None of the above. "
        rel_dict = {'A': 'Work_For', 'B': 'Kill', 'C': 'OrgBased_In', 'D': 'Live_In', 'E': 'Located_In', 'F': 'None'}

    else:
        raise ValueError("Invalid dataset")

    with open(args.path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.goldpath, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    for i in range(len(data)):
        for spoes in data[i]['uncertain_spo_list']:
            sub = spoes[0].replace(' ', '')
            ob = spoes[2].replace(' ', '')
            rel = spoes[1]
            if spoes[3] != []:
                for spo in spoes[3]:
                    counters = {}
                    answer = ''
                    stop_flag = False
                    while True and not stop_flag:
                        if spo == 'subject':
                            filled_prompt = sub_prompt.format(data[i]["text"], sub, ob, sub, sub, ob, rel)
                        elif spo == 'object':
                            filled_prompt = obj_prompt.format(data[i]["text"], sub, ob, ob, sub, ob, rel)
                        elif spo == 'predicate':
                            filled_prompt = rel_prompt.format(data[i]["text"], sub, ob)
                        else:
                            raise ValueError("Invalid spo")
                        rule_ = "You may only respond in the following format: enclose your answer options in parentheses. Only provide the parentheses and letters, without any other analytical content."
                        filled_prompt = filled_prompt + rule_
                        print(filled_prompt)
                        res = predict(filled_prompt)
                        print(res)
                        time.sleep(1)
                        results = detect_answer(res)
                        if results:
                            for result in results:
                                if result not in counters:
                                    counters[result] = 1
                                else:
                                    counters[result] += 1

                                if counters[result] >= 5:
                                    print(f"Result {result} reached 5 times, program stops.")
                                    stop_flag = True
                                    answer = result
                                    break
                    if spo == 'subject' or spo == 'object':
                        if answer == 'A' :
                            if [spoes[0], spoes[1], spoes[2]] not in gold[i]['pred_spo_list']:
                                gold[i]['pred_spo_list'].append([spoes[0], spoes[1], spoes[2]])
                            else:
                                print(f"already in gold no {i+1}")
                        elif answer == 'B' :
                            if [spoes[0], spoes[1], spoes[2]] in gold[i]['pred_spo_list']:
                                gold[i]['pred_spo_list'].remove([spoes[0], spoes[1], spoes[2]])
                            else:
                                print(f"already not in gold no {i+1}")
                        else:
                            print(f"no. {i+1} error")
                    elif spo == 'predicate':
                        if answer == list(rel_dict.keys())[-1]:
                            gold[i]['pred_spo_list'].remove([spoes[0], spoes[1], spoes[2]])
                        elif answer in list(rel_dict.keys()):
                            gold[i]['pred_spo_list'].remove([spoes[0], spoes[1], spoes[2]])
                            gold[i]['pred_spo_list'].append([spoes[0], rel_dict[answer], spoes[2]])
                        else:
                            print(f"no. {i+1} error ：{answer}")
    print(gold)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese", do_basic_tokenize=False, do_lower_case=True)
    calculate_f1(gold, tokenizer)

if __name__ == '__main__':
    main()


