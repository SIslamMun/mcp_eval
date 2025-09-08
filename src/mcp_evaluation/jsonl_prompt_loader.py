"""
JSONL-based prompt loader for reading prompts directly from JSONL datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Prompt metadata from JSONL entry."""
    
    id: int
    name: str
    complexity: str = Field(..., pattern="^(low|medium|high)$")
    category: str
    target_mcp: List[str]
    description: str
    timeout: Optional[int] = None
    expected_tools: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PromptData(BaseModel):
    """Complete prompt data from JSONL entry."""
    
    metadata: PromptMetadata
    content: str
    
    @classmethod
    def from_jsonl_entry(cls, data: Dict[str, Any]) -> "PromptData":
        """Create PromptData from JSONL entry."""
        content = data.pop("content", "")
        metadata = PromptMetadata(**data)
        return cls(metadata=metadata, content=content)


class JSONLPromptLoader:
    """
    Load prompts directly from JSONL files instead of individual .md files.
    
    This supports the professor's suggestion to use a single JSONL file
    for the dataset instead of many small .md files.
    """
    
    def __init__(self, jsonl_file: str = "prompts/prompts_dataset.jsonl"):
        """
        Initialize JSONL prompt loader.
        
        Args:
            jsonl_file: Path to JSONL file containing prompts
        """
        self.jsonl_file = Path(jsonl_file)
        self._prompt_cache: Dict[int, PromptData] = {}
        self._loaded = False
        
    def _load_jsonl(self) -> None:
        """Load all prompts from JSONL file into cache."""
        if self._loaded:
            return
            
        if not self.jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_file}")
            
        self._prompt_cache.clear()
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    prompt_data = PromptData.from_jsonl_entry(data)
                    self._prompt_cache[prompt_data.metadata.id] = prompt_data
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num} in {self.jsonl_file}: {e}")
        
        self._loaded = True
        print(f"📖 Loaded {len(self._prompt_cache)} prompts from {self.jsonl_file}")
    
    def load_prompt(self, prompt_id: int, inject_id: bool = True) -> PromptData:
        """
        Load prompt by ID from JSONL file.
        
        Args:
            prompt_id: Prompt identifier
            inject_id: Whether to inject prompt ID into content for tracking
            
        Returns:
            PromptData with metadata and content (with optional ID injection)
            
        Raises:
            FileNotFoundError: If JSONL file not found
            ValueError: If prompt ID not found
        """
        self._load_jsonl()
        
        if prompt_id not in self._prompt_cache:
            available_ids = sorted(self._prompt_cache.keys())
            raise ValueError(f"Prompt {prompt_id} not found. Available IDs: {available_ids}")
            
        prompt_data = self._prompt_cache[prompt_id]
        
        if inject_id:
            # Inject prompt ID as HTML comment at the beginning of content
            # This allows the processor to extract it later from database logs
            injected_content = f"<!-- EVAL_PROMPT_ID:{prompt_id} -->\n{prompt_data.content}"
            
            # Create a copy with injected content
            return PromptData(
                metadata=prompt_data.metadata,
                content=injected_content
            )
        
        return prompt_data
    
    def load_all_prompts(self) -> Dict[int, PromptData]:
        """
        Load all prompts from JSONL file.
        
        Returns:
            Dictionary mapping prompt IDs to PromptData
        """
        self._load_jsonl()
        return self._prompt_cache.copy()
    
    def get_prompts_by_complexity(self, complexity: str) -> List[PromptData]:
        """
        Get all prompts with specified complexity.
        
        Args:
            complexity: "low", "medium", or "high"
            
        Returns:
            List of matching PromptData
        """
        if complexity not in ["low", "medium", "high"]:
            raise ValueError("Complexity must be 'low', 'medium', or 'high'")
            
        self._load_jsonl()
        return [p for p in self._prompt_cache.values() if p.metadata.complexity == complexity]
    
    def get_prompts_by_category(self, category: str) -> List[PromptData]:
        """
        Get all prompts with specified category.
        
        Args:
            category: Category name (e.g., "jarvispipelinecreate")
            
        Returns:
            List of matching PromptData
        """
        self._load_jsonl()
        return [p for p in self._prompt_cache.values() if p.metadata.category == category]
    
    def get_prompts_by_mcp(self, mcp_name: str) -> List[PromptData]:
        """
        Get all prompts targeting specific MCP.
        
        Args:
            mcp_name: Name of MCP server
            
        Returns:
            List of matching PromptData
        """
        self._load_jsonl()
        return [p for p in self._prompt_cache.values() if mcp_name in p.metadata.target_mcp]
    
    def get_prompts_by_tags(self, tags: List[str]) -> List[PromptData]:
        """
        Get all prompts with any of the specified tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            List of matching PromptData
        """
        self._load_jsonl()
        matching = []
        
        for prompt in self._prompt_cache.values():
            if prompt.metadata.tags and any(tag in prompt.metadata.tags for tag in tags):
                matching.append(prompt)
                
        return matching
    
    def get_available_ids(self) -> List[int]:
        """Get list of available prompt IDs."""
        self._load_jsonl()
        return sorted(self._prompt_cache.keys())
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        self._load_jsonl()
        categories = set(p.metadata.category for p in self._prompt_cache.values())
        return sorted(categories)
    
    def get_available_complexities(self) -> Dict[str, int]:
        """Get complexity distribution."""
        self._load_jsonl()
        complexities = {"low": 0, "medium": 0, "high": 0}
        for prompt in self._prompt_cache.values():
            complexities[prompt.metadata.complexity] += 1
        return complexities
    
    def search_prompts(self, query: str, search_in: List[str] = None) -> List[PromptData]:
        """
        Search prompts by text query.
        
        Args:
            query: Search query
            search_in: Fields to search in ["name", "description", "content", "category", "tags"]
                      If None, searches in all fields
            
        Returns:
            List of matching PromptData
        """
        if search_in is None:
            search_in = ["name", "description", "content", "category", "tags"]
            
        self._load_jsonl()
        query_lower = query.lower()
        matching = []
        
        for prompt in self._prompt_cache.values():
            found = False
            
            if "name" in search_in and query_lower in prompt.metadata.name.lower():
                found = True
            elif "description" in search_in and query_lower in prompt.metadata.description.lower():
                found = True
            elif "content" in search_in and query_lower in prompt.content.lower():
                found = True
            elif "category" in search_in and query_lower in prompt.metadata.category.lower():
                found = True
            elif "tags" in search_in and prompt.metadata.tags:
                if any(query_lower in tag.lower() for tag in prompt.metadata.tags):
                    found = True
            
            if found:
                matching.append(prompt)
        
        return matching
    
    def validate_jsonl_file(self) -> Dict[str, Any]:
        """
        Validate JSONL file structure and content.
        
        Returns:
            Validation report with errors and statistics
        """
        report = {
            "file_path": str(self.jsonl_file),
            "file_exists": self.jsonl_file.exists(),
            "total_lines": 0,
            "valid_prompts": 0,
            "invalid_lines": 0,
            "errors": [],
            "duplicated_ids": [],
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "categories": set(),
            "mcp_targets": set(),
            "prompt_ids": []
        }
        
        if not self.jsonl_file.exists():
            report["errors"].append(f"JSONL file does not exist: {self.jsonl_file}")
            return report
        
        seen_ids = set()
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                report["total_lines"] += 1
                
                try:
                    data = json.loads(line.strip())
                    prompt_data = PromptData.from_jsonl_entry(data)
                    report["valid_prompts"] += 1
                    
                    # Check for duplicate IDs
                    if prompt_data.metadata.id in seen_ids:
                        report["duplicated_ids"].append(prompt_data.metadata.id)
                    else:
                        seen_ids.add(prompt_data.metadata.id)
                        
                    # Update statistics
                    report["complexity_distribution"][prompt_data.metadata.complexity] += 1
                    report["categories"].add(prompt_data.metadata.category)
                    report["mcp_targets"].update(prompt_data.metadata.target_mcp)
                    report["prompt_ids"].append(prompt_data.metadata.id)
                    
                except Exception as e:
                    report["invalid_lines"] += 1
                    report["errors"].append(f"Line {line_num}: {str(e)}")
        
        # Convert sets to sorted lists
        report["categories"] = sorted(list(report["categories"]))
        report["mcp_targets"] = sorted(list(report["mcp_targets"]))
        report["prompt_ids"] = sorted(report["prompt_ids"])
        
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded prompts."""
        self._load_jsonl()
        
        if not self._prompt_cache:
            return {"error": "No prompts loaded"}
        
        stats = {
            "total_prompts": len(self._prompt_cache),
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "categories": {},
            "mcp_targets": {},
            "tags": {},
            "id_range": {
                "min": min(self._prompt_cache.keys()),
                "max": max(self._prompt_cache.keys())
            }
        }
        
        for prompt in self._prompt_cache.values():
            # Complexity distribution
            stats["complexity_distribution"][prompt.metadata.complexity] += 1
            
            # Category distribution
            stats["categories"][prompt.metadata.category] = stats["categories"].get(prompt.metadata.category, 0) + 1
            
            # MCP target distribution
            for mcp in prompt.metadata.target_mcp:
                stats["mcp_targets"][mcp] = stats["mcp_targets"].get(mcp, 0) + 1
            
            # Tag distribution
            for tag in prompt.metadata.tags or []:
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
        
        return stats


# Compatibility layer to maintain existing interface
class UnifiedPromptLoader:
    """
    Unified loader that tries JSONL first, falls back to Markdown files.
    """
    
    def __init__(self, prompts_dir: str = "prompts", jsonl_file: str = None):
        self.prompts_dir = Path(prompts_dir)
        
        if jsonl_file is None:
            # Try common JSONL filenames
            possible_jsonl = [
                self.prompts_dir / "prompts_dataset.jsonl",
                self.prompts_dir / "dataset.jsonl",
                self.prompts_dir / "prompts.jsonl"
            ]
            
            for jsonl_path in possible_jsonl:
                if jsonl_path.exists():
                    jsonl_file = str(jsonl_path)
                    break
        
        self.use_jsonl = jsonl_file is not None
        
        if self.use_jsonl:
            self.jsonl_loader = JSONLPromptLoader(jsonl_file)
            print(f"📄 Using JSONL prompt source: {jsonl_file}")
        else:
            from .prompt_loader import MarkdownPromptLoader
            self.md_loader = MarkdownPromptLoader(str(self.prompts_dir))
            print(f"📝 Using Markdown prompt sources from: {self.prompts_dir}")
    
    def load_prompt(self, prompt_id: int, inject_id: bool = True) -> PromptData:
        """Load prompt by ID from preferred source."""
        if self.use_jsonl:
            return self.jsonl_loader.load_prompt(prompt_id, inject_id=inject_id)
        else:
            # Convert from old format
            old_prompt = self.md_loader.load_prompt(prompt_id)
            
            content = old_prompt.content
            if inject_id:
                # Inject prompt ID as HTML comment for tracking
                content = f"<!-- EVAL_PROMPT_ID:{prompt_id} -->\n{content}"
            
            return PromptData(
                metadata=PromptMetadata(
                    id=old_prompt.metadata.id,
                    name=f"{prompt_id:03d}-{old_prompt.metadata.complexity}-general",
                    complexity=old_prompt.metadata.complexity,
                    category="general",
                    target_mcp=old_prompt.metadata.target_mcp,
                    description=old_prompt.metadata.description,
                    timeout=old_prompt.metadata.timeout,
                    expected_tools=old_prompt.metadata.expected_tools,
                    tags=old_prompt.metadata.tags
                ),
                content=content
            )
    
    def load_all_prompts(self) -> Dict[int, PromptData]:
        """Load all prompts from preferred source."""
        if self.use_jsonl:
            return self.jsonl_loader.load_all_prompts()
        else:
            # Convert from old format
            old_prompts = self.md_loader.load_all_prompts()
            converted = {}
            
            for prompt_id, old_prompt in old_prompts.items():
                converted[prompt_id] = PromptData(
                    metadata=PromptMetadata(
                        id=old_prompt.metadata.id,
                        name=f"{prompt_id:03d}-{old_prompt.metadata.complexity}-general",
                        complexity=old_prompt.metadata.complexity,
                        category="general",
                        target_mcp=old_prompt.metadata.target_mcp,
                        description=old_prompt.metadata.description,
                        timeout=old_prompt.metadata.timeout,
                        expected_tools=old_prompt.metadata.expected_tools,
                        tags=old_prompt.metadata.tags
                    ),
                    content=old_prompt.content
                )
            
            return converted
    
    def get_available_ids(self) -> List[int]:
        """Get available prompt IDs."""
        if self.use_jsonl:
            return self.jsonl_loader.get_available_ids()
        else:
            return list(self.md_loader.load_all_prompts().keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded prompts."""
        if self.use_jsonl:
            return self.jsonl_loader.get_stats()
        else:
            # Return basic stats for MD source
            prompts = self.md_loader.load_all_prompts()
            return {
                "total_prompts": len(prompts),
                "source": "markdown",
                "prompt_ids": sorted(list(prompts.keys()))
            }
