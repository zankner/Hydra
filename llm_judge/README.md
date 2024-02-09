# LLM Judge
| [Original Github Repository](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

Our evaluation code and README is again taken from [Medusa](https://github.com/FasterDecoding/Medusa)

## Installation

| [Guide](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)

## Usage
To run the evaluation scripts on a Hydra model, run the following command (note that ```-model-id``` must contain the word "vicuna" for proper MT-Bench judging):

```bash
export CUDA_VISIBLE_DEVICES=0 # set the GPU id
python gen_model_answer_hydra.py  --model-path [HF repo or path] --model-id [ID of your choice] --base-model-id vicuna [Substitude if different base]
```

To run the baseline, run:

```bash
export CUDA_VISIBLE_DEVICES=0 # set the GPU id
python gen_model_answer_baseline.py  --model-path [HF repo or path] --model-id [ID of your choice] --base-model-id vicuna [Substitude if different base] 
```
(Please note we that we only support greedy decoding in these files.)


To query the results, run:

```
export OPENAI_API_KEY=$OPENAI_API_KEYs # set the OpenAI API key
python gen_judgement.py --model-list [model-id from above]
```

To obtain the results of the GPT-4 judge, run:

```
python show_result.py
```

## Citation
Please cite the original paper if you find the code or datasets helpful.
```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
