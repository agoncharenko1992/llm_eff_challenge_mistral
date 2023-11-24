# llm_eff_challenge_mistral

Here is brief introduction of my solution for llm efficient challenge. Unfortunately I didn't take some prize place only the 4-th one but I still hope that my findings will be useful for something.

**Strong baseline is all you need**
First of all we need select strong baseline. If we look at the open llm leaderboard (https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) one can see that Mistral-7b is the 
best model for selection as the baseline since it works like Llama2-13b.

**Datasets**
There was some candidates for tuning datasets:
1. Databricks Dolly
2. OpenPlatypus
3. Flan

Since efficient llm challenge has time restriction up to 24h we have to drop Flan.

**QLoRA Tuning**
The main technique to tune LLM is LoRA (https://arxiv.org/abs/2106.09685) and its quantized version QLoRA (https://arxiv.org/abs/2305.14314)
We need to add LoRA to all layers with possible rank reduction since it performs better than selective addition (https://twitter.com/Tim_Dettmers/status/1689375417189412864). 

**NEFTune**
The last one technique is NEFTune (https://arxiv.org/abs/2310.05914). It is sort of adversarial training when we add some noise to embeddings which boosts instruction finetuning. 
Fortunately there is no need to do it manually since it realized in trl package (you have to install in manually from github).
