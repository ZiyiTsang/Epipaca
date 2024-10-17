
 # The code implementation to Epipaca
![Epipaca](https://i.postimg.cc/qvtxQkWH/Epipaca.png "Epipaca")
This is the cross-languadge LLM adapter design for epilepsy-care instruction, with support both Mandarin and English.
- [Epipaca Checkpoint](https://huggingface.co/CocoNutZENG/Epipaca)
- [Synthetics Dataset](https://huggingface.co/datasets/CocoNutZENG/Epilepsy_Synthetics)
## Training steps
### Generate the seed_task(~200 Record)
In this step, we handwrite some of the task in seed_task, then we ask the LLM to generate more seed_task record in both Mandarin and English. The seed_task is the instruction for epilepsy-care. After that, we check the generated seed_task and remove the bad generated record.
### Generate the synthetic data(2k Record)
In this step, we ask the LLM to generate more synthetic data in both Mandarin and English. The man-write filter-rule is applied to filter the bad generated record.
We also upload the [Epilepsy_Synthetics](https://huggingface.co/datasets/CocoNutZENG/Epilepsy_Synthetics "Epilepsy_Synthetics") dataset for research proposes only. 
### Finetune the LLM
In this step, we finetune the LLM with the seed_task and synthetic data.  