# HarmMetric Eval

The file `benchmark_codes/scoring.py` contains the core code for scoring in our benchmark.

Our dataset can be found [here](https://huggingface.co/datasets/qusgo/HarmMetric_Eval).


## Abstract
The alignment of large language models (LLMs) with human values is critical for their safe deployment, yet jailbreak attacks can subvert this alignment to elicit harmful outputs from LLMs. In recent years, a proliferation of jailbreak attacks has emerged, accompanied by diverse metrics and judges to assess the harmfulness of the LLM outputs. However, the absence of a systematic benchmark to assess the quality and effectiveness of these metrics and judges undermines the credibility of the reported jailbreak effectiveness and other risks. To fill this gap, we introduce HarmMetric Eval benchmark, comprising a comprehensive dataset of harmful prompts paired with diverse harmful and non-harmful responses, along with a flexible measure designed to accommodate various metrics and judges. Our benchmark supports both overall and fine-grained effectiveness assessments, enabling a nuanced analysis of the strengths and weaknesses of current harmfulness metrics. With HarmMetric Eval, our extensive experiments uncover a surprising result: *two conventional metrics—METEOR and ROUGE-1—outperform LLM-based judges in evaluating the harmfulness of model responses, challenging prevailing beliefs about LLMs' superiority in this domain.*

![alt text](pipeline.png)


## Quick Start

The following is only the evaluation process for **GCG_eval** and **AutoDAN_eval**.  

You can modify the `eval_list` at the beginning of `metrics_codes/eval_with_metrics.py` to select the **harmfulness metric** you wish to evaluate, or use your own metric. However, remember to update the model paths or OpenAI API keys for the corresponding methods in `metrics_codes/curr_metrics.py` to your own.  

You can build environment for all metrics by following the step below, except for **Shen_eval**, which requires downgrading `transformers` to **4.27.1**:  

```bash
pip install -r requirements.txt
```

If you only want to try **GCG_eval** and **AutoDAN_eval** and score them using *HarmMetric Eval*, the following command is sufficient to set up the required environment:

```bash
pip install pandas==2.2.3
```

### Experimental replication

Follow the steps below, and you will find the detailed evaluation results of **GCG_eval** and **AutoDAN_eval** on our dataset in the `temp_results` folder.

```bash
cd metrics_codes
python eval_with_metrics.py
cd ..
```

### Scoring metrics

Follow the steps below, and you will be able to see the **overall** and **fine-grained effectiveness scores** of **GCG_eval** and **AutoDAN_eval** on our benchmark in the `results` folder.

```bash
cd benchmark_codes
python scoring.py --metric all
cd ..
```




