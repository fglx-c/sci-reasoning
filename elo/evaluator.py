import asyncio
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Handle both relative and absolute imports
try:
    from ..utils.llm_utils import LLMClient, create_llm_client
    from ..utils.text_utils import extract_tagged_content
except ImportError:
    # Fallback for direct execution - add parent directory to path
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from utils.llm_utils import LLMClient, create_llm_client
    from utils.text_utils import extract_tagged_content

from .config import ELOConfig, DEFAULT_CONFIG


def get_judge_idea_all_prompt(idea0: str, idea1: str, topic: str) -> str:
    """Adapt from CoI-Agent prompt for idea comparison."""
    prompt = f"""You are a judge in a competition. You have to decide which idea is better.

The idea0 is: {idea0}

The idea1 is: {idea1}

The topic is: {topic}

Which idea do you think is better? Please write a short paragraph to explain your choice.

Here are your evaluation criteria:
1. Novelty: Are the problems or approaches new? Is this a novel combination of familiar techniques? Is it clear how this work differs from previous contributions? Is related work adequately referenced? 
2. Significance: Are the idea important? Are other people (practitioners or researchers) likely to use these ideas or build on them? Does the idea address a difficult problem in a better way than previous research? Does it provide a unique theoretical or pragmatic approach?
3. Feasibility: Can the idea be realized with existing technology or methods? Are there any technical difficulties or bottlenecks? Is the idea clear and logical? Is there any obvious error or unreasonable part in the idea, and can the experiment be designed normally according to this idea. 
4. Clarity: Is the paper clearly written? Is it well-organized? Does it adequately inform the reader? 
5. Effectiveness: How likely the proposed idea is going to work well (e.g., better than existing baselines).

Note: 
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. DO NOT allow the LENGTH of the responses to influence your evaluation, choose the one that is straight-to-the-point instead of unnecessarily verbose. Be as objective as possible. (very important!!!)

If you think idea0 is better than idea1, you should output 0. If you think idea1 is better than idea0, you should output 1. If you think idea0 and idea1 are equally good, you should output 2.

Your output should be strictly in following format:
Your thinking process:
...

Your choice:
<novelty>{{{{ Your choice for novelty }}</novelty>
<significance>{{{{ Your choice for significance }}</significance>
<feasibility>{{{{ Your choice for feasibility }}</feasibility>
<clarity>{{{{ Your choice for clarity }}</clarity>
<effectiveness>{{{{ Your choice for effectiveness }}</effectiveness>"""
    return prompt


@dataclass
class EvaluationItem:
    """Container for items to be evaluated."""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""
    item_a_id: str
    item_b_id: str
    scores: Dict[str, int]  # criterion -> score (0, 1, or 2)
    raw_response: str
    timestamp: datetime


@dataclass
class ELOResult:
    """Final ELO evaluation results."""
    rankings: List[Tuple[str, float]]  # (item_id, elo_score)
    comparisons: List[ComparisonResult]
    metadata: Dict[str, Any]
    
    def get_best_item(self) -> Tuple[str, float]:
        """Get the highest-ranked item."""
        return self.rankings[0] if self.rankings else (None, 0)


class ELOEvaluator:
    """ELO evaluation system using original CoI-Agent approach."""
    
    def __init__(self, config: ELOConfig = None):
        """Initialize ELO evaluator."""
        self.config = config or DEFAULT_CONFIG
        self.llm_client = create_llm_client(model=self.config.model)
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        if self.config.save_results:
            os.makedirs(self.config.save_directory, exist_ok=True)
    
    async def evaluate_batch(self, items: List[EvaluationItem], context: str = "") -> ELOResult:
        """Evaluate a batch of items using ELO tournament."""
        if len(items) < 2:
            raise ValueError("Need at least 2 items for comparison")
        
        self.logger.info(f"Starting ELO evaluation of {len(items)} items")
        
        # Initialize ELO scores (same as CoI-Agent)
        elo_scores = [self.config.initial_elo_score for _ in range(len(items))]
        
        # Generate all pairwise comparisons
        comparison_pairs = []
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    comparison_pairs.append((i, j))
        
        # Run comparisons
        comparison_results = await self._run_comparisons(comparison_pairs, items, context)
        
        # Update ELO scores (same logic as CoI-Agent)
        elo_scores = self._update_elo_scores(elo_scores, comparison_results)
        
        # Create rankings
        rankings = sorted(
            [(items[i].id, elo_scores[i]) for i in range(len(items))],
            key=lambda x: x[1], reverse=True
        )
        
        result = ELOResult(
            rankings=rankings,
            comparisons=comparison_results,
            metadata={
                "evaluation_time": datetime.now().isoformat(),
                "num_items": len(items),
                "num_comparisons": len(comparison_results),
                "context": context
            }
        )
        
        # Save results
        if self.config.save_results:
            await self._save_results(result)
        
        return result
    
    async def _run_comparisons(self, pairs: List[Tuple[int, int]], 
                             items: List[EvaluationItem], context: str) -> List[ComparisonResult]:
        """Run pairwise comparisons."""
        if self.config.parallel_evaluations:
            # Process in batches
            batch_size = self.config.max_concurrent_evaluations
            results = []
            
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_tasks = [
                    self._compare_pair(items[i], items[j], context)
                    for i, j in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Comparison failed: {result}")
                    else:
                        results.append(result)
                
                if i + batch_size < len(pairs):
                    await asyncio.sleep(1)
            
            return results
        else:
            # Sequential processing
            results = []
            for i, j in pairs:
                try:
                    result = await self._compare_pair(items[i], items[j], context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Comparison failed: {e}")
            return results
    
    async def _compare_pair(self, item_a: EvaluationItem, item_b: EvaluationItem,
                          context: str) -> ComparisonResult:
        """Perform single pairwise comparison"""
        prompt = get_judge_idea_all_prompt(item_a.content, item_b.content, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm_client.generate_response_async(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        if not response:
            response = "Error: Could not get LLM response"
        
        # Extract scores using original criteria
        scores = {}
        criteria = ["novelty", "significance", "feasibility", "clarity", "effectiveness"]
        
        for criterion in criteria:
            score_text = extract_tagged_content(response, criterion, fallback_to_full_text=False)
            try:
                score = int(score_text.strip())
                if score in [0, 1, 2]:
                    scores[criterion] = score
                else:
                    scores[criterion] = 2  # Default to tie
            except (ValueError, AttributeError):
                scores[criterion] = 2  # Default to tie on parse error
        
        return ComparisonResult(
            item_a_id=item_a.id,
            item_b_id=item_b.id,
            scores=scores,
            raw_response=response,
            timestamp=datetime.now()
        )
    
    def _update_elo_scores(self, elo_scores: List[float], 
                          results: List[ComparisonResult]) -> List[float]:
        """Update ELO scores"""
        updated_scores = elo_scores.copy()
        
        def change_winner_to_score(winner, score_1, score_2):
            """Original CoI-Agent scoring function."""
            try:
                winner = int(winner)
            except:
                return score_1 + 0.5, score_2 + 0.5
            if winner == 0:
                return score_1 + 1, score_2
            if winner == 2:
                return score_1 + 0.5, score_2 + 0.5
            return score_1, score_2 + 1
        id_to_index = {}
        for i, (item_id, _) in enumerate([(f"item_{j}", 0) for j in range(len(elo_scores))]):
            id_to_index[item_id] = i
        
        for result in results:
            i = id_to_index.get(result.item_a_id, 0)
            j = id_to_index.get(result.item_b_id, 0)
        
            for criterion, winner in result.scores.items():
                updated_scores[i], updated_scores[j] = change_winner_to_score(
                    winner, updated_scores[i], updated_scores[j]
                )
        
        return updated_scores
    
    async def _save_results(self, result: ELOResult):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elo_evaluation_{timestamp}.json"
        filepath = os.path.join(self.config.save_directory, filename)
        
        result_dict = {
            "rankings": result.rankings,
            "comparisons": [
                {
                    "item_a_id": comp.item_a_id,
                    "item_b_id": comp.item_b_id,
                    "scores": comp.scores,
                    "raw_response": comp.raw_response,
                    "timestamp": comp.timestamp.isoformat()
                }
                for comp in result.comparisons
            ],
            "metadata": result.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)


# Convenience function
async def evaluate_ideas(ideas: List[str], topic: str = "", config: ELOConfig = None) -> ELOResult:
    """Convenience function to evaluate a list of ideas."""
    eval_items = [
        EvaluationItem(id=f"item_{i}", content=idea) 
        for i, idea in enumerate(ideas)
    ]
    
    evaluator = ELOEvaluator(config or DEFAULT_CONFIG)
    return await evaluator.evaluate_batch(eval_items, topic) 