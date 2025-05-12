import json
import time
import curr_metrics
import os

eval_list = [
            "GCG_eval", 
            "AutoDAN_eval",

            # "squad_f1_score",
            # "cos_sim", 
            # "BERT_sim", 
            # "meteor", 
            # "bleu", 
            # "rouge1",
            # "rouge2", 
            # "rougeL", 
            # "rougeLsum",

            # "JbB_eval",
            # "PAIR_eval",
            # "Qi_eval",
            # "llama_guard_1_eval",
            # "llama_guard_2_eval",
            # "llama_guard_3_eval",

            # "HarmBench_eval", 
            # "GPTF_eval",

            # "Shen_eval", # you may need to change the version of transformers
            ]


key_list = ["harmful_prompt",  
            "harmful_res_0", 
            "harmful_res_1",
            "harmful_res_2", 
            "harmful_res_3",
            "irr_res_0", 
            "irr_res_1", 
            "irr_res_2", 
            "refusal", 
            "prevention",  
            "redirection", 
            "useless_aff"
            ]


infile_path = "../dataset/harmful_prompt_and_res.jsonl"
all_scores = {}
    

for blk in eval_list:

    if blk == "GPTF_eval":
        model, tokenizer, device = curr_metrics.GPTF_init()
    if blk == "HarmBench_eval":
        model, tokenizer = curr_metrics.HarmBench_init()
    if blk == "llama_guard_1_eval":
        model, tokenizer = curr_metrics.llama_guard_init("/hub/huggingface/models/meta/LlamaGuard-7b")
    if blk == "llama_guard_2_eval":
        model, tokenizer = curr_metrics.llama_guard_init("/hub/huggingface/models/meta/Llama-Guard-2-8B")
    if blk == "llama_guard_3_eval":
        model, tokenizer = curr_metrics.llama_guard_init("/hub/huggingface/models/meta/Llama-Guard-3-8B")
    if blk == "Shen_eval":
        model, tokenizer = curr_metrics.Shen_init()



    def get_function_by_name(func_name):
        if hasattr(curr_metrics, func_name):
            return getattr(curr_metrics, func_name)
        else:
            raise ValueError(f"Function {func_name} not found in my_module!")

    f_name = blk
    if "rouge" in f_name:
        f_name = "rouge"
    if "llama_guard" in f_name:
        f_name = "llama_guard_eval"
        
    f_eval = get_function_by_name(f_name)


    if not os.path.exists("../temp_results"):
        os.makedirs("../temp_results")
    outfile_path = f"../temp_results/{blk}.jsonl"


    def try_again(func:callable, string0:str, string1:str):
        for i in range(2):
            ret = func(string0, string1)
            if ret < 0:
                print(f"Ret: {ret}. Retrying...")
                time.sleep(1)
                continue
            break
        return ret
    

    with open(infile_path, 'r', encoding='utf-8') as infile, open(outfile_path, 'w', encoding='utf-8') as outfile:
        idx = 0
        fail_num = 0
        for line in infile:
            scores = {}
            new_data = {}
            # if idx >= 3:
            #     break
            idx += 1
            print(blk + " " + str(idx))

            data = json.loads(line.strip())
            items = {}

            prompt = data.get('harmful_prompt')
            ref_res_0 = data.get('ref_res_0')
            ref_res_1 = data.get('ref_res_1')
            qid = data.get('id')

            for key in key_list:
                items[key] = data.get(key)
                if items[key] == '' or items[key] is None:
                    print(f"{key} not exist!")
                    exit(0)

            if blk in ["GCG_eval", "AutoDAN_eval"]:
                for key in key_list:
                    scores[key] = f_eval(items[key])

            elif "rouge" in blk:
                for key in key_list:
                    scores[key] = (f_eval(ref_res_0, items[key], blk) + f_eval(ref_res_1, items[key], blk))/2.0

            elif blk in ["squad_f1_score", "cos_sim", "meteor", "bleu", "BERT_sim"]:
                for key in key_list:
                    scores[key] = (f_eval(ref_res_0, items[key]) + f_eval(ref_res_1, items[key]))/2.0

            elif blk in ["GPTF_eval"]:
                for key in key_list:
                    scores[key] = f_eval(model, tokenizer, device, items[key])

            elif blk in ["HarmBench_eval"]:
                for key in key_list:
                    scores[key] = f_eval(model, tokenizer, prompt, items[key])

            elif blk in ["Qi_eval", 
                        "PAIR_eval", 
                        "JbB_eval",
                        "Shen_eval",
                        "llama_guard_1_eval",
                        "llama_guard_2_eval",
                        "llama_guard_3_eval", 
                        ]:
                
                if "llama_guard" in blk or blk == "Shen_eval":
                    prompt = [model, tokenizer, prompt]

                temp_scores = {}
                total_round = len(key_list)
                fail_flag = False

                for round in range(3):
                    temp_scores[round] = {}
                    i = 0
                    print('\r'+' '*24, end="", flush=True)
                    for key in key_list:
                        i += 1
                        temp_scores[round][key] = try_again(f_eval, prompt, items[key])
                        if temp_scores[round][key] < 0:
                            fail_flag = True
                            break
                        print(f"\r{round}: {i}/{total_round}", end="", flush=True)
                    if fail_flag == True:
                        break
                if fail_flag == True:
                        print("Fail!")
                        fail_num += 1
                        continue

                for key in key_list:
                    scores[key] = (temp_scores[0][key] + temp_scores[1][key] + temp_scores[2][key])/3.0
                print('')


            new_data = {}
            new_data['id'] = qid
            new_data['scores'] = scores
            
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')
            outfile.flush()

        print(f'fail_num of {blk}: {fail_num}\n')
