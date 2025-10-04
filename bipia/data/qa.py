# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from bipia.data.utils import embed_blurb

from .base import QAPIABuilder


class QAIPIADataset(QAPIABuilder):
    name = "qa"
    system_prompt = (
        "You are a question-and-answer assistant responsible for assisting the user in processing and analyzing news content. "
        "Please fulfill the user's requests based on the content of the news. "
        "Your answers should be concise and meet the user's requirements.\n"
        f"{embed_blurb}"
    )
    user_prompt_template = (
        "Please answer my question based on the content of the news:\n<|begin_embed|>{context}<|end_embed|>\n\n"
        "Your answers should be as concise as possible.\n\n"
        "Question: {question}"
    )

    def construct_prompt(self, example: Any) -> tuple[str, str]:
        # if require_system_prompt:
        #     system_prompt = self.system_prompt.format(
        #         context=example["context"], guidance=ign_guidance
        #     )
        #     user_prompt = self.user_prompt_template[0].format(
        #         question=example["question"]
        #     )
        #     return system_prompt, user_prompt
        # else:
        #     user_prompt = self.user_prompt_template[1].format(
        #         context=example["context"],
        #         question=example["question"],
        #         guidance=ign_guidance,
        #     )
        #     return user_prompt
        return ("TODO", "TODO")

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"][0]
        return f"Answer: {ideal}."
