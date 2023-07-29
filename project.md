ssh mchorse@216.153.51.21

- Data collection
- Exploratory data analysis
  - Reward function analysis
- Influence measurement
  - Topic modeling
  - Encoding prediction
  - Perplexity scoring
- Reward modeling
- Inverse reinforcement learning

- What do philosophers value?
- How does it change over time?
- How does it change across philosophers?


1. Embed abstracts
2. Calculate influence
3. Train reward model over all years
4. Identify most influential papers, people, and topics
5. Does this improve MMLU, ETHICS, and other metrics?

Given prior knowledge {t-1}
Which paper from {t} predicts {t+1}, or which when averaged into knowledge makes {t-1} look like {t+1}?


---
1. Base model
   1. Take PhilPapers 2020 survey
   2. Compare results to human answers
2. Reward model
   1. Calculate reward for abstracts by similarity to 2021 abstracts
   2. Train reward model
   3. Retake survey
3. RLHF model
   1. Train RLHF model against reward model
   2. Retake survey


Median cosine similarity between old and new abstracts in cluster, 1/k