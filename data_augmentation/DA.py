import os
import argparse
import random
import openai
openai.api_key = ''
import time

parser = argparse.ArgumentParser(description='hello')
parser.add_argument('--input_folder', type=str, default='input_eng', help='input folder')
parser.add_argument('--attributes', type=str, default='attributes.txt', help='attribute file')
parser.add_argument('--num_of_output', type=int, default=3, help='number of output')
parser.add_argument('--output_file', type=str, default='output.jsonl', help='output file')
args = parser.parse_args()

def predict(prompt):
    retry_count = 100
    retry_interval = 1
    for _ in range(retry_count):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "NLP professor especilly in specific domain"},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            msg = response.choices[0].message["content"].strip()
            return msg
        except openai.error.RateLimitError as e:
            print("Exceeded OpenAI API call rate.", e)
            print('Please retry...')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
        except Exception as e:
            print("Task execution error:", e)
            print('Please retry...')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)

def main(prompt):
    attr_ = []
    sub_ = []
    with open(args.input_folder+ "/" + args.attributes, 'r', encoding='utf-8') as f:
        for line in f:
            attr_.append(line.strip().split(' ')[0])
            sub_.append(line.strip().split(' ')[1])

    print(attr_)
    print(sub_)

    data = {f'{attr}': [] for attr in attr_}

    for i in range(len(attr_)):
        with open(args.input_folder+ "/" + attr_[i] + "/" + attr_[i] + ".txt", 'r', encoding='utf-8') as f:
            for line in f:
                data[attr_[i]].append(line.strip())

    count = 0
    start_time = time.time()
    while(count < args.num_of_output):
        for i in range(len(data)):
            random.shuffle(data[attr_[i]])

        subdata = {f'{sub}': [] for sub in sub_ if sub != 'not'}
        for i in range(len(data)):
            if sub_[i] == "not":
                continue
            else:
                with open(args.input_folder+ "/" + attr_[i] + "/" + data[attr_[i]][0] + "/" + sub_[i] + ".txt", 'r',
                          encoding='utf-8') as f:
                    for line in f:
                        subdata[sub_[i]].append(line.strip())

        for i in range(len(sub_)):
            if sub_[i] == "not":
                continue
            else:
                random.shuffle(subdata[sub_[i]])

        example = []
        with open(args.input_folder + "/example/example.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                example.append(line.strip())

        random.shuffle(example)


        prompt = prompt.format(data["fault_component"][0], subdata["fault_phenomenon"][0], data["context"][0], example[0],
                               example[1])
        print("\n")
        print(prompt)
        res = predict(prompt)
        time.sleep(1)
        print('chatgpt answer: ', res)
        count += 1
        print(str(count) + "/" + str(args.num_of_output))
        nowtime = time.time()
        print('run_time: {} s'.format(round(nowtime - start_time, 3)))
        if not os.path.exists("output/" + args.input_folder):
            os.makedirs("output/" + args.input_folder)
        with open("output/" + args.input_folder + "/" + args.output_file, 'a', encoding='utf-8') as f:
            f.write(res + '\n')

    end_time = time.time()
    total_run_time = round(end_time - start_time, 3)
    print('Total_run_time: {} s'.format(total_run_time))

if __name__ == '__main__':
    prompt = "You are an domain automotive industry’s expert. Your task is to write data for the automotive industry fault dataset (for a training set for relational triple extraction), please follow these requirements: \
            The data should be aimed at sub-domain in the automotive industry fault. \n\
            The fault component of the text is {}, which should be the subject of the triple. \
            The fault phenomenon of the text is {}, which should be the object of the triple. \
            The predicate of the triple should be part_failure (don't have to mention it in text).\
            The context of the text is {}. The context can be more diverse and realistic but should not include any other components of faults. \
            The text length should be in 20-50 words. \n\
            Here are some examples: \
            {}{}\n\
            The format of the generated data should strictly follow the format (jsonl) in the example.  The text in text and spo_list must maintain consistent capitalization. \n\
            Do not add any extra information on your own, as this may introduce noise. Must not mention any other fault and component. There is also no need to mention what consequences it led to."
    main(prompt)
