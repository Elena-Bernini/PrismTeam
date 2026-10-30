# 03 Partial layout retrieval

Creates incomplete graph queries by masking rooms and retrieves compatible complete layouts. It depends only on stage 01 processed layout/graph JSONs.

Configs include random 30% masking, remove-bedroom, and remove-bathroom experiments. Safe smoke-test override: `--max-items 1000 --num-queries 10 --candidate-pool 100`.
