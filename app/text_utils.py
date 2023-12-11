import json
import os
import re
from typing import List, Union

pattern = r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|ninceteen|twenty|\d+)\s+"

preamble = r'(show me|give me|show|give|generate|create|create for me)\s+'
as_pattern = r'\ba\b\s+(\w+)'


def clean_prompt(prompt: str, remove_a_s: bool = False) -> str:
    prompt = prompt.lower()
    prompt = re.sub(preamble, "", prompt, flags=re.IGNORECASE)

    if remove_a_s:
        prompt = re.sub(as_pattern, r"\1", prompt, flags=re.IGNORECASE, count=1)

    return prompt


numbers = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty"
]


def split_prompt(prompt: str, task_base_path: Union[str, None]) -> List[str]:
    _prompt = clean_prompt(prompt)
    prompts = _prompt.split(" and ")

    tasks = []
    for prompt in prompts:
        prompt = prompt.replace(",", "").replace(".", "").replace("?", "").replace("!", "")

        _match = re.search(pattern, prompt)
        if _match:
            try:
                num = int(_match.group(1))
            except ValueError:
                if numbers.index(_match.group(1)) > -1:
                    num = numbers.index(_match.group(1))
                else:
                    num = 1

            prompt = re.sub(pattern, "", prompt)
            prompt = prompt.strip()
        else:
            num = 1

        num = max(1, min(num, 20))
        tasks.extend([prompt] * num)

    if task_base_path:
        with open(os.path.join(task_base_path, "text.json"), 'w+') as jf:
            json.dump(
                dict(
                    prompt=_prompt,
                    split=prompts,
                    tasks=tasks
                ), jf)
        jf.close()

    return tasks


if __name__ == "__main__":
    text = "give me 7 penguins"
    print(split_prompt(text, None))
