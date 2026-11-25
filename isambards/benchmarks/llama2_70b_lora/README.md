# Isambards: Llama 2 70B LoRA

Source, container digest, launcher and system settings are preserved in source-or-patches/. config.sh is the configuration used for these ten runs.

To reproduce, restore data/ and the checker .venv inside source-or-patches/ as described in its README.md, then run ./run_and_time.sh. It submits ten runs with distinct seeds.

The official package checker must pass. Its normalization is saved in scaling.json. If the successful checker requires no normalization, the launcher writes scaling_factor=1.0.
