"""
Markdown prompt loader for structured prompts with YAML frontmatter.
"""

import frontmatter
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Prompt metadata from YAML frontmatter."""
    
    id: int
    complexity: str = Field(..., pattern="^(low|medium|high)$")
    target_mcp: List[str]
    description: str
    timeout: Optional[int] = None
    expected_tools: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class PromptData(BaseModel):
    """Complete prompt data with metadata and content."""
    
    metadata: PromptMetadata
    content: str
    file_path: Optional[str] = None


class MarkdownPromptLoader:
    """
    Load prompts from Markdown files with YAML frontmatter.
    
    Expected format:
    ---
    id: 1
    complexity: "low"
    target_mcp: ["mcp-name"]
    description: "Brief description"
    expected_tools: ["Bash", "Edit"]
    tags: ["basic", "testing"]
    ---
    # Prompt Title
    
    Your prompt text here...
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._prompt_cache: Dict[int, PromptData] = {}
        
    def load_prompt(self, prompt_id: int) -> PromptData:
        """
        Load prompt by ID.
        
        Args:
            prompt_id: Prompt identifier
            
        Returns:
            PromptData with metadata and content
            
        Raises:
            FileNotFoundError: If prompt file not found
            ValueError: If prompt format is invalid
        """
        # Check cache first
        if prompt_id in self._prompt_cache:
            return self._prompt_cache[prompt_id]
            
        # Find prompt file
        prompt_file = self._find_prompt_file(prompt_id)
        if not prompt_file:
            raise FileNotFoundError(f"Prompt {prompt_id} not found in {self.prompts_dir}")
            
        # Load and parse prompt
        prompt_data = self._parse_prompt_file(prompt_file)
        
        # Validate ID matches
        if prompt_data.metadata.id != prompt_id:
            raise ValueError(
                f"Prompt ID mismatch: file has {prompt_data.metadata.id}, requested {prompt_id}"
            )
            
        # Cache and return
        self._prompt_cache[prompt_id] = prompt_data
        return prompt_data
    
    def load_all_prompts(self) -> Dict[int, PromptData]:
        """
        Load all prompts from directory.
        
        Returns:
            Dictionary mapping prompt IDs to PromptData
        """
        prompts = {}
        
        # Find all markdown files
        if not self.prompts_dir.exists():
            return prompts
            
        for prompt_file in self.prompts_dir.glob("*.md"):
            try:
                prompt_data = self._parse_prompt_file(prompt_file)
                prompts[prompt_data.metadata.id] = prompt_data
            except Exception as e:
                print(f"Warning: Failed to load {prompt_file}: {e}")
                
        return prompts
    
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
            
        all_prompts = self.load_all_prompts()
        return [p for p in all_prompts.values() if p.metadata.complexity == complexity]
    
    def get_prompts_by_mcp(self, mcp_name: str) -> List[PromptData]:
        """
        Get all prompts targeting specific MCP.
        
        Args:
            mcp_name: Name of MCP server
            
        Returns:
            List of matching PromptData
        """
        all_prompts = self.load_all_prompts()
        return [p for p in all_prompts.values() if mcp_name in p.metadata.target_mcp]
    
    def get_prompts_by_tags(self, tags: List[str]) -> List[PromptData]:
        """
        Get all prompts with any of the specified tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            List of matching PromptData
        """
        all_prompts = self.load_all_prompts()
        matching = []
        
        for prompt in all_prompts.values():
            if prompt.metadata.tags and any(tag in prompt.metadata.tags for tag in tags):
                matching.append(prompt)
                
        return matching
    
    def _find_prompt_file(self, prompt_id: int) -> Optional[Path]:
        """Find prompt file by ID."""
        if not self.prompts_dir.exists():
            return None
            
        # Try common naming patterns
        patterns = [
            f"{prompt_id:03d}.md",
            f"prompt_{prompt_id:03d}.md",
            f"prompt_{prompt_id}.md",
            f"{prompt_id}.md",
        ]
        
        for pattern in patterns:
            prompt_file = self.prompts_dir / pattern
            if prompt_file.exists():
                return prompt_file
                
        # Search all files for matching ID
        for prompt_file in self.prompts_dir.glob("*.md"):
            try:
                prompt_data = self._parse_prompt_file(prompt_file)
                if prompt_data.metadata.id == prompt_id:
                    return prompt_file
            except Exception:
                continue
                
        return None
    
    def _parse_prompt_file(self, prompt_file: Path) -> PromptData:
        """Parse prompt file with frontmatter."""
        try:
            # Load file with frontmatter
            with open(prompt_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                
            # Validate metadata
            metadata = PromptMetadata(**post.metadata)
            
            # Get content (strip extra whitespace)
            content = post.content.strip()
            
            return PromptData(
                metadata=metadata,
                content=content,
                file_path=str(prompt_file)
            )
            
        except Exception as e:
            raise ValueError(f"Failed to parse {prompt_file}: {e}")
    
    def create_sample_prompt(self, prompt_id: int, output_dir: Optional[str] = None) -> str:
        """
        Create a sample prompt file for reference.
        
        Args:
            prompt_id: ID for the sample prompt
            output_dir: Directory to create file in (defaults to prompts_dir)
            
        Returns:
            Path to created file
        """
        if output_dir is None:
            output_dir = self.prompts_dir
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_content = f"""---
id: {prompt_id}
complexity: "low"
target_mcp: ["example-mcp"]
description: "Sample prompt for testing MCP functionality"
expected_tools: ["Bash", "Read"]
tags: ["sample", "testing"]
---

# Sample MCP Evaluation Prompt

This is a sample prompt for evaluating MCP (Model Context Protocol) functionality.

"""

        sample_file = output_dir / f"{prompt_id:03d}.md"
        sample_file.write_text(sample_content)
        
        return str(sample_file)
    
    def validate_prompt_directory(self) -> Dict[str, Any]:
        """
        Validate all prompts in directory.
        
        Returns:
            Validation report with errors and statistics
        """
        report = {
            "total_files": 0,
            "valid_prompts": 0,
            "invalid_prompts": 0,
            "errors": [],
            "duplicated_ids": [],
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "mcp_targets": set(),
            "prompt_ids": []
        }
        
        if not self.prompts_dir.exists():
            report["errors"].append(f"Prompt directory {self.prompts_dir} does not exist")
            return report
        
        seen_ids = set()
        
        for prompt_file in self.prompts_dir.glob("*.md"):
            report["total_files"] += 1
            
            try:
                prompt_data = self._parse_prompt_file(prompt_file)
                report["valid_prompts"] += 1
                
                # Check for duplicate IDs
                if prompt_data.metadata.id in seen_ids:
                    report["duplicated_ids"].append(prompt_data.metadata.id)
                else:
                    seen_ids.add(prompt_data.metadata.id)
                    
                # Update statistics
                report["complexity_distribution"][prompt_data.metadata.complexity] += 1
                report["mcp_targets"].update(prompt_data.metadata.target_mcp)
                report["prompt_ids"].append(prompt_data.metadata.id)
                
            except Exception as e:
                report["invalid_prompts"] += 1
                report["errors"].append(f"{prompt_file.name}: {str(e)}")
        
        # Convert set to sorted list
        report["mcp_targets"] = sorted(list(report["mcp_targets"]))
        report["prompt_ids"] = sorted(report["prompt_ids"])
        
        return report
