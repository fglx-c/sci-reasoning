# Pairwise Verifier Usage Guide

This document explains how to use the new pairwise ranking verifier in your GRPO training.

## Quick Start

To switch from the regular verifier to pairwise ranking, simply change the strategy in your config:

### Configuration Changes

1. **Edit your config file** (e.g., `verl/trainer/config/ppo_trainer.yaml`):

```yaml
reward_model:
  enable: True
  strategy: pairwise  # Changed from 'verifier' to 'pairwise'
  model:
    path: /path/to/your/comparison/model  # Can be same as verifier model
    # ... other model settings
  pairwise_config:
    comparison_temperature: 0.0  # Use deterministic comparisons
    max_tokens: 10              # Short responses for A/B decisions
    cache_size: 10000           # Cache comparison results
    retry_on_invalid: true      # Retry if response isn't A/B
```

2. **Run training** with the same command:

```bash
./start_grpo_ray_job.sh --model_name Qwen-2.5-7B
```

That's it! The system will automatically use pairwise ranking instead of individual scoring.

## How It Works

### Algorithm Overview

1. **Group by Prompt**: Samples with same UID are grouped together for ranking
2. **Random Start**: One sample is randomly selected as initial ranking
3. **Binary Search Insertion**: Remaining samples are inserted via pairwise comparisons
4. **Linear Rewards**: Final ranking is converted to linear rewards (best=1.0, worst=0.0)

### Comparison Process

For each pairwise comparison, the LLM receives:

```
Question: What is 2 + 2?
Ground Truth Answer: 4

Response A: The answer is 4.
Response B: 2 + 2 equals four.

Compare these responses considering accuracy, reasoning quality, and clarity.
Which response is better?

Answer: A or B
```

### Reward Assignment

Final rewards are linearly distributed:
- **Best response**: 1.0
- **Worst response**: 0.0  
- **Middle responses**: Evenly spaced between 0 and 1

Example with 4 samples: `[1.0, 0.67, 0.33, 0.0]`

## Configuration Options

### Pairwise Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `comparison_temperature` | 0.0 | Temperature for comparison LLM (0.0 = deterministic) |
| `max_tokens` | 10 | Maximum tokens for A/B responses |
| `cache_size` | 10000 | Number of comparisons to cache |
| `retry_on_invalid` | true | Retry if response is not A/B |

### Model Configuration

The comparison model can be:
- **Same as verifier**: Use the same model path as your existing verifier
- **Different model**: Use a specialized comparison model
- **Any chat model**: Any model that can follow comparison instructions

## Performance Characteristics

### Comparison Complexity

- **Per group**: O(n log n) comparisons where n = group size
- **Example**: 8 samples per group ≈ 24 comparisons
- **Caching**: Reduces redundant comparisons significantly

### Memory Usage

- **Cache**: Configurable size (default 10,000 comparisons)
- **Model**: Same GPU memory as regular verifier
- **Overhead**: Minimal compared to model inference

## Switching Between Strategies

You can easily switch between reward strategies:

### Individual Scoring (Original)
```yaml
reward_model:
  strategy: verifier
```

### Pairwise Ranking (New)
```yaml
reward_model:
  strategy: pairwise
  pairwise_config:
    # ... pairwise settings
```

Both strategies:
- Output the same `rm_scores` format
- Work with existing GRPO algorithm
- Require no code changes in training scripts

## Expected Benefits

### Advantages of Pairwise Ranking

1. **More Robust**: Relative comparisons vs absolute scoring
2. **Better Calibration**: Rankings are naturally normalized within groups
3. **Handling Ambiguity**: Easier to compare than to score absolutely
4. **Reduced Bias**: Less sensitive to model scoring quirks

### When to Use Pairwise

- **Multiple samples per prompt** (n > 1 in rollout)
- **Subjective quality measures** (style, reasoning quality)
- **Inconsistent absolute scoring** from your verifier
- **Want more stable reward distributions**

## Troubleshooting

### Common Issues

1. **"No rm_scores found"**: Check that `reward_model.enable: True`
2. **Invalid comparisons**: Model not following A/B format → enable `retry_on_invalid`
3. **Slow training**: Reduce `cache_size` or use faster comparison model
4. **Memory issues**: Reduce rollout group size or use smaller comparison model

### Debug Mode

To see what comparisons are being made, you can enable debug logging:

```bash
export VERL_PPO_LOGGING_LEVEL=DEBUG
```

This will show comparison prompts and results in the logs.

## Example Training Command

```bash
# Using pairwise ranking with Qwen-2.5-7B
./start_grpo_ray_job_nvidia_small_test.sh \
  --model_name Qwen-2.5-7B \
  --rollout_n 4 \
  --total_epochs 3
```

The system will automatically use pairwise ranking if configured in the YAML file. 