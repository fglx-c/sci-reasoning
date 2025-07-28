from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class ELOConfig:
    """Configuration for ELO evaluation system."""
    
    # LLM Configuration
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # ELO Tournament Settings
    initial_elo_score: float = 1200.0
    k_factor: float = 32.0
    
    # Evaluation Settings
    parallel_evaluations: bool = True
    max_concurrent_evaluations: int = 5
    
    # Output
    save_results: bool = True
    save_directory: str = "elo_results"
    
    @classmethod
    def from_env(cls) -> "ELOConfig":
        """Create config from environment variables."""
        return cls(
            model=os.environ.get("MAIN_LLM_MODEL", "gpt-4"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4000")),
            parallel_evaluations=os.environ.get("ELO_PARALLEL", "true").lower() == "true",
            max_concurrent_evaluations=int(os.environ.get("ELO_MAX_CONCURRENT", "5")),
            save_directory=os.environ.get("ELO_SAVE_DIR", "elo_results")
        )


# Default configuration
DEFAULT_CONFIG = ELOConfig() 