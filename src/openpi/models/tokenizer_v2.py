import logging

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        try:
            # Decode predicted output tokens
            decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
            
            # Debug: print decoded tokens for troubleshooting
            print(f"[DEBUG] Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"[DEBUG] Decoded: '{decoded_tokens[:100]}{'...' if len(decoded_tokens) > 100 else ''}'")

            # Extract actions from FAST model outputs
            if "Action: " not in decoded_tokens:
                print(f"[WARNING] No 'Action: ' found in decoded tokens. Using zero actions.")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)

            # Extract actions from decoded tokens
            raw_action_tokens = np.array(
                self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
            )
            action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
            
            # Try different time_horizons if the first attempt fails
            for th in [action_horizon, 8, 6, 4, 2, 1]:
                try:
                    result = self._fast_tokenizer.decode(
                        [action_tokens.tolist()], time_horizon=th, action_dim=action_dim
                    )[0]
                    
                    print(f"[DEBUG] Successfully decoded with time_horizon={th}, shape={result.shape}")
                    
                    # Adjust to expected shape if needed
                    if result.shape[0] != action_horizon:
                        result = self._adjust_action_horizon(result, action_horizon)
                        print(f"[DEBUG] Adjusted to shape={result.shape}")
                    
                    return result
                    
                except Exception as e:
                    if th == action_horizon:  # Only print error for the first attempt
                        print(f"[DEBUG] Failed with time_horizon={th}: {e}")
                    continue
            
            # If all attempts failed, return zero actions
            print(f"[ERROR] All decode attempts failed. Returning zero actions.")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
            
        except Exception as e:
            print(f"[ERROR] extract_actions failed: {e}")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
    
    def _adjust_action_horizon(self, actions: np.ndarray, target_horizon: int) -> np.ndarray:
        """Adjust action sequence length to match target horizon."""
        current_horizon = actions.shape[0]
        
        if current_horizon == target_horizon:
            return actions
        elif current_horizon > target_horizon:
            # Truncate
            return actions[:target_horizon]
        else:
            # Pad by repeating the last action
            last_action = actions[-1:] if current_horizon > 0 else np.zeros((1, actions.shape[1]))
            padding = np.tile(last_action, (target_horizon - current_horizon, 1))
            return np.vstack([actions, padding])

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
