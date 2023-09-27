# WiCE: Real-World Entailment for Claims in Wikipedia

Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, Greg Durrett

This repository contains the dataset for "[WiCE: Real-World Entailment for Claims in Wikipedia](https://arxiv.org/pdf/2303.01432.pdf)".

## WiCE

`data` directory includes the WiCE dataset.

### Entailment and Retrieval

`data/entailment_retrieval` includes the WiCE dataset for entailment and retrieval task. `data/entailment_retrieval/claim` includes data with the original claims and `data/entailment_retrieval/subclaims` includes data with the decomposed claims (finegrained annotation by using Claim-Split).

Each sub-directory includes jsonl files for train, dev, and test sets. Here is an example of the data in the jsonl files:

```json
{
    "label": "partially_supported",
    "supporting_sentences": [[5, 15], [15, 17]],
    "claim": "Arnold is currently the publisher and editorial director of Media Play News, one of five Hollywood trades and the only one dedicated to the home entertainment sector.",
    "evidence": [list of evidence sentences],
    "meta": {"id": "dev02986", "claim_title": "Roger Hedgecock", "claim_section": "Other endeavors.", "claim_context": [paragraph]}
}
```

* `label`: Entailment label in {`supported`, `partially_supported`, `not_supported`}
* `supporting_sentences`: List of indices of supporting sentences. All provided sets of supporting sentences are valid (in the above example, both `[5, 15]` and `[5, 17]` are annotated as correct sets of supporting sentences that include same information).
* `claim`: A sentence from Wikipedia
* `evidence`: A list of sentences in the cited website
* `meta`
  * `claim_title`: Title of the Wikipedia page that includes `claim`
  * `claim_section`: Section that includes `claim`
  * `claim_context`: Sentences just before `claim`

### Non-Supported Tokens

`data/non_supported_tokens` includes the WiCE dataset for non-supported tokens detection task. We only provide annotation for sub-claims that are annotated as `partially_supported`. We filtered out data points with low inter-annotator agreement (please refer to the paper for details).

```json
{
    "claim": "Irene Hervey appeared in over fifty films and numerous television series.",
    "claim_tokens": ["Irene", "Hervey", "appeared", "in", "over", "fifty", "films", "and", "numerous", "television", "series", "."],
    "non_supported_spans": [false, false, false, false, true, true, false, false, false, false, false, false],
    "evidence": [list of evidence sentences],
    "meta": {"id": "test00561-1", "claim_title": "Irene Hervey", "claim_section": "Abstract.", "claim_context": " Irene Hervey was an American film, stage, and television actress."}
}
```

* `claim_tokens`: List of tokens in the claim
* `non_supported_spans`: List of bool corresponding to `claim_tokens` (`true` is non-supported tokens)

## Claim-Split

`claim_split` directory includes prompts for Claim-Split, a method to decompose claims by using GPT-3. We use different prompts for different datasets in the experiments in this work, so we provide prompts for WiCE, VitaminC, PAWS, and FRANK (XSum).

![](figures/claim_split.png)

## License

The WiCE dataset is based on Wikipedia articles and websites archived at Common Crawl. The majority of text content in Wikipedia is licensed under the [Creative Commons Attribution Share-Alike license](https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License) (CC-BY-SA). For more information about the Wikipedia policy, please refer to [this page](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content). You are also bound by the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use) when using this dataset.

Our annotations are released under the terms of [ODC-BY](https://opendatacommons.org/licenses/by/1-0/).
