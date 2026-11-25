"""Managed hosts can replace the skills-index guidance and drop the Hermes help pointer.

A host whose users never configure the agent themselves (Diadems) needs neither the
"help with Hermes Agent itself" pointer nor the built-in "you MUST load a skill even
partially relevant" rule: config.yaml ``agent.hermes_help_guidance: false`` and
``skills.index_guidance: <text>``.
"""

from agent import prompt_builder
from agent.prompt_builder import (
    HERMES_AGENT_HELP_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS,
    _render_skills_index,
)
from tests.agent.test_system_prompt import _make_agent, _prompt_parts

SKILLS = {"creation": [("report", "Write a report"), ("slides", "Make slides")]}


def test_built_in_guidance_is_kept_by_default():
    text = _render_skills_index(SKILLS, None, None)
    assert "you MUST load it" in text
    assert "Only proceed without loading a skill" in text


def test_index_guidance_replaces_the_built_in_text_and_keeps_every_skill():
    text = _render_skills_index(SKILLS, None, None, guidance="Load a skill when it fits the task.")
    assert text.startswith("## Skills\nLoad a skill when it fits the task.\n\n<available_skills>\n")
    assert "- report: Write a report" in text and "- slides: Make slides" in text
    assert "MUST" not in text and "Only proceed without loading" not in text
    assert text.endswith("</available_skills>")


def test_index_guidance_is_read_from_config(monkeypatch):
    monkeypatch.setattr("agent.skill_utils._skills_cfg_get", lambda key: {"index_guidance": "  Use skills.  "}.get(key))
    assert prompt_builder._skills_index_guidance() == "Use skills."
    monkeypatch.setattr("agent.skill_utils._skills_cfg_get", lambda key: ["not", "text"])
    assert prompt_builder._skills_index_guidance() == ""


def test_help_pointer_is_present_by_default():
    stable = _prompt_parts(_make_agent())["stable"]
    assert HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS in stable


def test_help_pointer_can_be_switched_off():
    stable = _prompt_parts(_make_agent(_hermes_help_guidance=False))["stable"]
    assert HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS not in stable
    assert HERMES_AGENT_HELP_GUIDANCE not in stable
    assert "hermes-agent.nousresearch.com/docs" not in stable
