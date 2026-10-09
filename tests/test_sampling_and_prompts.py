"""Regression checks that do not require CityFlow, GPUs, or an LLM server."""

import ast
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time
import unittest
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]


def load_sampling_test():
    """Load the real sampling method without importing simulator dependencies."""
    path = ROOT / "framework/FTSample.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FTS")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "test")
    namespace = {
        "time": time,
        "deepcopy": deepcopy,
        "defaultdict": defaultdict,
        "tqdm": lambda values: values,
        "action2code": lambda signal: signal,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["test"]


class SamplingAndPromptTests(unittest.TestCase):
    def test_each_intersection_selects_using_its_own_state(self):
        class InitialActionsSelected(Exception):
            pass

        # Stop at the simulator checkpoint, after the real initial action loop.
        for count in (1, 3):
            with self.subTest(intersections=count):
                states = [{"intersection": index} for index in range(count)]
                agents = [MagicMock() for _ in states]
                sampler = MagicMock()
                sampler.dic_traffic_env_conf = {"RUN_COUNTS": 30, "MIN_ACTION_TIME": 30}
                sampler.env.get_current_time.return_value = 0
                sampler.process_state.return_value = (states, [])
                sampler.find_trajectory.return_value = ({}, {})
                sampler.agent_intersection_list = agents
                sampler.history_states = []
                sampler.long_info = False
                sampler.estimate_effective_range.side_effect = lambda state: [state["intersection"]]
                sampler.save_env.side_effect = InitialActionsSelected

                with self.assertRaises(InitialActionsSelected):
                    load_sampling_test()(sampler, None, 0)

                self.assertEqual(
                    [call.args[0] for call in sampler.estimate_effective_range.call_args_list],
                    states,
                )
                for index, agent in enumerate(agents):
                    agent.select_signal_default.assert_called_once_with([index])

    def test_all_west_left_turn_prompt_definitions_point_north(self):
        definitions = []
        for directory in ("framework", "models", "utils"):
            for path in sorted((ROOT / directory).glob("*.py")):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if "WL is the West Left-turn lane" in node.value:
                            definitions.append((path.relative_to(ROOT), node.lineno, node.value))

        self.assertTrue(definitions, "No WL prompt definitions found")
        for path, line, prompt in definitions:
            with self.subTest(path=str(path), line=line):
                self.assertIn("traffic turns left from west to north", prompt)
                self.assertNotIn("traffic turns left from west to south", prompt)


if __name__ == "__main__":
    unittest.main()
