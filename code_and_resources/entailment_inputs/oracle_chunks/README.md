## Oracle Chunks

The oracle retrieval dataset simulates the situation that you have a perfect evidence retrieval model. Each row of the dataset includes a entailment label, claim, and retrieved chunk ("evidence") in the following format:

```json
{
    "label": "partially_supported",
    "claim": "Arnold is currently the publisher and editorial director of Media Play News, one of five Hollywood trades and the only one dedicated to the home entertainment sector.",
    "evidence": ["Media Play News", "Media Play News is the voice of the home entertainment industry.", "We reach major studios, independent suppliers, technology companies such as Microsoft and Roku and a growing number of distributors of digital content.", "Thomas K. Arnold, Publisher and Editorial Director, Media Play News:", "For more than 12 years he was the Publisher and Editorial Director of Home Media Magazine, the home entertainment industry's weekly trade publication.", "He joined Video Store Magazine, Home Media's predecessor, in 1991.", "She spearheaded the publication's reviews section, as well as aggressive coverage of the home video sales market.", "She also helped launch the magazine's Web site in 1996.", "In her position as editor-in-chief since 2006, she has spearheaded the launch of such projects as the daily blast, transmitted via email each day to readers, and Agent DVD, a consumer publication aimed at genre enthusiasts who attend Comic-Con International in San Diego.", "John Boezinger, Associate Publisher and Advertising Director, Media Play News:", "Before joining Home Media Magazine in September 2007 as an Account Executive, Boezinger worked for Advanstar Communications, Home Media Magazine's former parent company, for 10 years.", "He worked in sales management as well as in inside and outside sales for three different vertical industry groups: Automotive, Powersports and CRM.", "Boezinger graduated from the University of Michigan in Ann Arbor and has an MBA from California State University at Long Beach.", "As Executive Editor he oversees editorial production of our monthly print and digital editions and also has oversight of the market research department."],
    "meta": {
        "id": "dev02986",
        "chunk_idx": [2, 5, 10, 15, 17, 18, 26, 27, 28, 31, 33, 34, 35, 38],  # indices of sentences included in the chunk (including dummy sentences)
        "oracle_idx": [5, 15]  # indices of actual supporting sentences
    }
}
```

The sentences in `oracle_idx` are the actual supporting sentences provided by an annotator. To avoid the biases caused by the number of supporting sentences (e.g. partially supported cases may have smaller number of supporting sentences), we add randomly selected sentences to oracle chunks (`evidence`). The indices of the sentences included in the oracle chunk (`evidence`) are in `chunk_idx`.

There are multiple rows (3 for most cases) for the same claim/subclaim (same id) with different retrieved (oracle) chunks, because different annotators can annotate different gold supporting sentences. We assume that you take tha maximum entailment score over all chunks for each claim/subclaim to get the final score. However, you may also use other strategies such as taking the average score over all chunks.

[../../code/exec_files/run_dataset_preprocessing.sh](../../code/exec_files/run_dataset_preprocessing.sh) includes code for generating the oracle retrieval dataset from the original WiCE dataset. [../../code/exec_files/evaluate_gpt_on_oracle.sh](../../code/exec_files/evaluate_gpt_on_oracle.sh) includes code for evaluating GPT-3.5 and GPT-4 on the oracle chunks. You can use similar code for evaluating your models on the oracle chunks.

When you report the result on this data, you need to clearly mention that you use the oracle retrieval dataset, not the original WiCE dataset.
