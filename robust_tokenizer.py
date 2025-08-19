#!/usr/bin/env python3

import sys
import traceback
import numpy as np
from openpi.models.tokenizer import FASTTokenizer

class RobustFASTTokenizer(FASTTokenizer):
    """带有错误处理的FAST Tokenizer"""
    
    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        try:
            # 原始的extract_actions逻辑
            decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
            
            # 记录解码结果
            print(f"[DEBUG] Decoded tokens: '{decoded_tokens}'")
            
            if "Action: " not in decoded_tokens:
                print(f"[WARNING] No 'Action: ' found in decoded tokens. Using fallback.")
                return self._generate_fallback_actions(action_horizon, action_dim)
            
            # 提取动作
            raw_action_tokens = np.array(
                self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
            )
            action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
            
            # 尝试解码，如果失败则使用不同的time_horizon
            for th in [action_horizon, 8, 6, 4, 2, 1]:
                try:
                    result = self._fast_tokenizer.decode(
                        [action_tokens.tolist()], time_horizon=th, action_dim=action_dim
                    )[0]
                    
                    print(f"[DEBUG] Successfully decoded with time_horizon={th}, shape={result.shape}")
                    
                    # 如果shape不匹配，调整到期望的shape
                    if result.shape[0] != action_horizon:
                        result = self._adjust_action_horizon(result, action_horizon)
                        print(f"[DEBUG] Adjusted to shape={result.shape}")
                    
                    return result
                    
                except Exception as e:
                    print(f"[DEBUG] Failed with time_horizon={th}: {e}")
                    continue
            
            # 如果所有尝试都失败，使用fallback
            print(f"[ERROR] All decode attempts failed. Using fallback.")
            return self._generate_fallback_actions(action_horizon, action_dim)
            
        except Exception as e:
            print(f"[ERROR] extract_actions failed: {e}")
            traceback.print_exc()
            return self._generate_fallback_actions(action_horizon, action_dim)
    
    def _adjust_action_horizon(self, actions: np.ndarray, target_horizon: int) -> np.ndarray:
        """调整action序列长度"""
        current_horizon = actions.shape[0]
        
        if current_horizon == target_horizon:
            return actions
        elif current_horizon > target_horizon:
            # 截断
            return actions[:target_horizon]
        else:
            # 填充：重复最后一个action
            last_action = actions[-1:] if current_horizon > 0 else np.zeros((1, actions.shape[1]))
            padding = np.tile(last_action, (target_horizon - current_horizon, 1))
            return np.vstack([actions, padding])
    
    def _generate_fallback_actions(self, action_horizon: int, action_dim: int) -> np.ndarray:
        """生成fallback actions（可以是零动作或维持当前状态的动作）"""
        # 可以根据需要返回零动作或其他安全动作
        return np.zeros((action_horizon, action_dim), dtype=np.float32)

# 测试函数
def test_robust_tokenizer():
    tokenizer = RobustFASTTokenizer(max_len=256)
    
    # 测试problematic tokens
    problematic_tokens = np.array([495, 319, 493, 283, 672, 363, 578, 421, 277, 429, 2021, 493, 578, 358, 357, 274, 579, 356, 257, 448])
    
    result = tokenizer.extract_actions(problematic_tokens, action_horizon=10, action_dim=7)
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")

if __name__ == "__main__":
    test_robust_tokenizer()
