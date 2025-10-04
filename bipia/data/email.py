# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from bipia.data.utils import embed_blurb
from .base import QAPIABuilder


class EmailIPIABuilder(QAPIABuilder):
    name = "email"
    system_prompt = (
        "You are an email assistant responsible for assisting the user in processing and analyzing emails. "
        "Please fulfill the user's requests based on the content of the email. "
        "Your answers should be concise and meet the user's requirements.\n"
        f"{embed_blurb}"
    )
    user_prompt_template = (
        "Please answer my question based on the content of this email:\n<|begin_embed|>{context}<|end_embed|>\n\n"
        "Question: {question}"
    )

    def construct_prompt(self, example: Any) -> Any:
        system_prompt = self.system_prompt
        user_prompt = self.user_prompt_template.format(
            context=example["context"], question=example["question"].lstrip("Q: ")
        )
        return system_prompt, user_prompt

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"]

        if ideal != "unkown":
            return f"Answer: {ideal}."
        else:
            return "There is no information about the question in the email."
