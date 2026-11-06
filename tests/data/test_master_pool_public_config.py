"""CPU-only CLI configuration checks; no FVQA data is loaded or generated."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_web_agent.data.quality.pool_builder import MasterPoolBuildConfig


class StopBeforeDataBuild(Exception):
    pass


class PublicMasterPoolConfigTests(unittest.TestCase):
    def test_public_configs_reach_the_builder_with_valid_schemas(self):
        spec = importlib.util.spec_from_file_location(
            "public_pool_cli", ROOT / "scripts/build_validated_master_pool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cases = (
            ("v0_1", "validated-master-pool-v0.1", "v0_3", "protocol-sft-v0.4"),
            ("v0_2", "validated-master-pool-v0.2", "v0_4", "protocol-sft-v0.5"),
        )
        for version, schema, previous, protocol in cases:
            with self.subTest(version=version):
                config_path = "configs/data_quality/validated_master_pool_%s_server.yaml" % version
                # main() performs real YAML loading and gate-config validation.
                # Stop at the data boundary so no raw data or outputs are touched.
                with patch.object(
                    sys, "argv",
                    ["build_validated_master_pool.py", "--project-root", str(ROOT),
                     "--config", config_path, "--output-dir",
                     "data/config_preflight_never_written"],
                ), patch.object(
                    cli, "build_validated_master_pool", side_effect=StopBeforeDataBuild
                ) as build, patch.object(cli, "write_master_pool") as write:
                    with self.assertRaises(StopBeforeDataBuild):
                        cli.main()
                    write.assert_not_called()
                build.assert_called_once()
                kwargs = build.call_args.kwargs
                config = kwargs["config"]
                config.validate()
                self.assertEqual(config.schema_version, schema)
                self.assertEqual(config.gate_schema_version, "shared-executability-gate-v0.2")
                self.assertEqual(config.protocol_sft_schema, protocol)
                self.assertEqual(
                    Path(config.previous_trajectory_path),
                    ROOT / "data/processed" / ("protocol_sft_" + previous) / "trajectories.jsonl",
                )
                self.assertEqual(kwargs["source_path"], ROOT / "data/raw/fvqa/fvqa_train.parquet")
                self.assertEqual(kwargs["cache_path"], ROOT / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl")
                self.assertEqual(len(config.gate_config_sha256), 64)

    def test_unsupported_historical_gate_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "gate schema mismatch"):
            MasterPoolBuildConfig(
                source_label="fixture", cache_label="fixture",
                gate_schema_version="shared-executability-gate-v0.1",
            ).validate()


if __name__ == "__main__":
    unittest.main()
