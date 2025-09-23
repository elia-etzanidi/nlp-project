# nlp-project
university project for natural language processing class

## Description
A text reconstruction toolkit combining **Grammatical Error Correction (GEC)** and **sentence-level paraphrasing (seq2seq models)**.  
Evaluation is split into two phases:  
- **Phase 1:** linguistic/structural metrics (GPT-2 Perplexity, ROUGE-L Recall, Distinct-n, etc.)  
- **Phase 2:** semantic shift analysis using **Word2Vec embeddings + cosine similarity** with visualization (**PCA/t-SNE**).

## Pipelines
- **Pipeline A:** LanguageTool → T5-GEC → PEGASUS  
- **Pipeline B:** T5-GEC → FLAN-T5  
- **Pipeline C:** LanguageTool → T5-GEC → BART

## Models Used
- Grammar correction: `vennify/t5-base-grammar-correction`  
- Paraphrasing: `tuner007/pegasus_paraphrase`, `eugenesiow/bart-paraphrase`  
- Rewriting: `google/flan-t5-large`  
- Fluency metric: **GPT-2** (perplexity)  
- Rule-based grammar checks: **LanguageTool**  
- Embeddings (Phase 2): **gensim Word2Vec** (trained ad hoc on reference + reconstructed texts)

## Requirements
- **Python 3.11.13**  
- **Poetry** for dependency management
