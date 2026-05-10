---
name: card
description: Generate a standardized model card documenting the trained model — type, performance, training data, limitations, intended use, and artifact contract.
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob
---

You generate a standardized model card from the experiment log, model contract, and config.

## Steps

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the model card generator:**
   ```bash
   python scripts/generate_model_card.py --config config.yaml --log experiments/log.jsonl --contract model_contract.md --output MODEL_CARD.md
   ```

3. **Read and present the generated card:**
   - Read `MODEL_CARD.md` and display it to the user.
   - If no experiments exist yet, inform the user and show the skeleton card.

4. **Suggest next steps:**
   - Review the **Ethical Considerations** section and fill in bias, fairness, and impact notes.
   - Review the **Intended Use** section and document what the model is NOT intended for.
   - If limitations mention overfitting, suggest running `/turing:validate` for stability checks.
   - If the card looks complete, suggest committing it to version control.

## Error Handling

- If `config.yaml` is missing, tell the user to run `/turing:init` first.
- If `experiments/log.jsonl` is missing or empty, generate a skeleton card and note that training is needed.
- If `.venv` doesn't exist, try `python3 scripts/generate_model_card.py` directly.
