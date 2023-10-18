GPT_PROMPT = """Your task is to evaluate if a claim is supported by a provided evidence article snippet.

We provide several examples. Your response must be in the same format as the XML in the examples.

Examples:
<input>
    <claim>On August 22, 2017, Richard Amardi was selected to play for the Canadian Senior Men's National Team to compete in the FIBA AmeriCup 2017 in Argentina.</claim>
    <evidence>
        <sentence_39>Values, Vision and Missions</sentence_39>
        <sentence_40># SENIOR MEN'S NATIONAL TEAM ANNOUNCES FIBA AMERICUP 2017 ROSTER</sentence_40>
        <sentence_41>TORONTO, ON (August 22, 2017) - Canada Basketball's Men's High Performance Program is pleased to announce the final roster for the Senior Men's National Team set to compete at the FIBA AmeriCup 2017 in Argentina.</sentence_41>
        <sentence_47>"The FIBA AmeriCup 2017 will be a great first step for our program towards our ultimate goal of qualifying for the Tokyo 2020 Olympic Games," said Jay Triano, Senior Men's National Team Program Head Coach.</sentence_47>
        <sentence_49>The FIBA AmeriCup 2017 will be played from August 25 to September 3, 2017 and brings together America's best 12 men's national teams.</sentence_49>
        <sentence_53>The Final Phase will be played in Cordoba (Argentina), where the winners of the semifinals will square off in the title game.</sentence_53>
        <sentence_54>Following the FIBA AmeriCup 2017, Canada's pursuit of a FIBA Basketball World Cup 2019 berth continues with the Americas Qualifiers where 16 teams will compete for seven World Cup spots.</sentence_54>
        <sentence_64>Thursday, August 24 (Air Canada Centre)1:45 p.m.</sentence_64>
        <sentence_69>2017 SMNT AMERICUP ROSTER</sentence_69>
        <sentence_73>Chalons-Reims (France)</sentence_73>
        <sentence_77>Richard Amardi</sentence_77>
        <sentence_80>Murphy Burnatowski</sentence_80>
    </evidence>
</input>
<!-- Your explanation and answer should be written below -->
<output>
    <explanation>Sentence 41 says that the final roster for the Senior Men's National Team set to compete at the FIBA AmeriCup 2017 in Argentina was announced on August 22, 2017. Sentence 77 shows that Richard Amardi is on the list.</explanation>
    <answer>supported</answer>
</output>

<input>
    <claim>On August 22, 2017, Richard Amardi was selected to play for the Canadian Senior Men's National Team to compete in the FIBA AmeriCup 2017 in Argentina.</claim>
    <evidence>
        <sentence_39>Values, Vision and Missions</sentence_39>
        <sentence_40># SENIOR MEN'S NATIONAL TEAM ANNOUNCES FIBA AMERICUP 2017 ROSTER</sentence_40>
        <sentence_47>"The FIBA AmeriCup 2017 will be a great first step for our program towards our ultimate goal of qualifying for the Tokyo 2020 Olympic Games," said Jay Triano, Senior Men's National Team Program Head Coach.</sentence_47>
        <sentence_49>The FIBA AmeriCup 2017 will be played from August 25 to September 3, 2017 and brings together America's best 12 men's national teams.</sentence_49>
        <sentence_53>The Final Phase will be played in Cordoba (Argentina), where the winners of the semifinals will square off in the title game.</sentence_53>
        <sentence_54>Following the FIBA AmeriCup 2017, Canada's pursuit of a FIBA Basketball World Cup 2019 berth continues with the Americas Qualifiers where 16 teams will compete for seven World Cup spots.</sentence_54>
        <sentence_64>Thursday, August 24 (Air Canada Centre)1:45 p.m.</sentence_64>
        <sentence_69>2017 SMNT AMERICUP ROSTER</sentence_69>
        <sentence_73>Chalons-Reims (France)</sentence_73>
        <sentence_77>Richard Amardi</sentence_77>
        <sentence_80>Murphy Burnatowski</sentence_80>
    </evidence>
</input>
<!-- Your explanation and answer should be written below -->
<output>
    <explanation>Sentence 40, 53, and 54 shows that this article inclues the Canadian Senior Men's National Team to compete in the FIBA AmeriCup 2017 in Argentina. Sentence 77 shows that Richard Amardi is on the list. However, the evidence does not include information about when this was announced.</explanation>
    <answer>partially_supported</answer>
</output>

<input>
    <claim>On August 22, 2017, Richard Amardi was selected to play for the Canadian Senior Men's National Team to compete in the FIBA AmeriCup 2017 in Argentina.</claim>
    <evidence>
        <sentence_39>Values, Vision and Missions</sentence_39>
        <sentence_40># SENIOR MEN'S NATIONAL TEAM ANNOUNCES FIBA AMERICUP 2017 ROSTER</sentence_40>
        <sentence_47>"The FIBA AmeriCup 2017 will be a great first step for our program towards our ultimate goal of qualifying for the Tokyo 2020 Olympic Games," said Jay Triano, Senior Men's National Team Program Head Coach.</sentence_47>
        <sentence_49>The FIBA AmeriCup 2017 will be played from August 25 to September 3, 2017 and brings together America's best 12 men's national teams.</sentence_49>
        <sentence_53>The Final Phase will be played in Cordoba (Argentina), where the winners of the semifinals will square off in the title game.</sentence_53>
        <sentence_54>Following the FIBA AmeriCup 2017, Canada's pursuit of a FIBA Basketball World Cup 2019 berth continues with the Americas Qualifiers where 16 teams will compete for seven World Cup spots.</sentence_54>
        <sentence_64>Thursday, August 24 (Air Canada Centre)1:45 p.m.</sentence_64>
        <sentence_69>2017 SMNT AMERICUP ROSTER</sentence_69>
        <sentence_73>Chalons-Reims (France)</sentence_73>
        <sentence_80>Murphy Burnatowski</sentence_80>
    </evidence>
</input>
<!-- Your explanation and answer should be written below -->
<output>
    <explanation>This article is about the Canadian Senior Men's National Team to compete in the FIBA AmeriCup 2017. However, the claim is not supported by the evidence.</explanation>
    <answer>not_supported</answer>
</output>

Here is your task:

<input>
    <claim>{claim}</claim>
    <evidence>
{evidence}
    </evidence>
</input>
<!-- Your explanation and answer should be written below -->"""


def get_gpt_prompt(claim: str, evidence_list: list[str], line_idx: list[int]) -> str:
    assert len(evidence_list) == len(line_idx), f"{len(evidence_list)} != {len(line_idx)}, {line_idx}"
    
    evidence_string = "\n".join([f"        <sentence_{idx}>{line}</sentence_{idx}>" for idx, line in zip(line_idx, evidence_list)])
    
    return GPT_PROMPT.format(claim=claim, evidence=evidence_string)
