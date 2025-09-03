"""
Test prompt loader functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from mcp_evaluation.prompt_loader import (
    MarkdownPromptLoader, 
    PromptMetadata, 
    PromptData
)


class TestPromptMetadata:
    """Test PromptMetadata validation."""
    
    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = PromptMetadata(
            id=1,
            complexity="low",
            target_mcp=["mcp-server"],
            description="Test prompt"
        )
        
        assert metadata.id == 1
        assert metadata.complexity == "low"
        assert metadata.target_mcp == ["mcp-server"]
        assert metadata.description == "Test prompt"
    
    def test_invalid_complexity(self):
        """Test invalid complexity validation."""
        with pytest.raises(ValueError):
            PromptMetadata(
                id=1,
                complexity="invalid",
                target_mcp=["mcp-server"],
                description="Test prompt"
            )
    
    def test_optional_fields(self):
        """Test optional fields."""
        metadata = PromptMetadata(
            id=1,
            complexity="medium",
            target_mcp=["mcp-server"],
            description="Test prompt",
            expected_tools=["Bash", "Edit"],
            tags=["test", "sample"]
        )
        
        assert metadata.timeout is None  # timeout is optional now
        assert metadata.expected_tools == ["Bash", "Edit"]
        assert metadata.tags == ["test", "sample"]


class TestMarkdownPromptLoader:
    """Test MarkdownPromptLoader functionality."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = MarkdownPromptLoader("test_prompts")
        assert str(loader.prompts_dir) == "test_prompts"
        assert loader._prompt_cache == {}
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    @patch('frontmatter.load')
    def test_find_prompt_file(self, mock_frontmatter, mock_file, mock_glob, mock_exists):
        """Test finding prompt files by different naming patterns."""
        loader = MarkdownPromptLoader("test_prompts")
        
        # Mock directory exists
        mock_exists.return_value = True
        
        # Mock file exists for specific pattern
        def exists_side_effect(path_obj):
            return str(path_obj).endswith("001.md")
        
        with patch.object(Path, 'exists', side_effect=exists_side_effect):
            result = loader._find_prompt_file(1)
            assert result is not None
            assert str(result).endswith("001.md")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('frontmatter.load')
    def test_parse_prompt_file(self, mock_frontmatter, mock_file):
        """Test parsing prompt file with frontmatter."""
        # Mock frontmatter data
        mock_post = type('MockPost', (), {
            'metadata': {
                'id': 1,
                'complexity': 'low',
                'target_mcp': ['test-mcp'],
                'description': 'Test prompt'
            },
            'content': 'This is the prompt content'
        })()
        mock_frontmatter.return_value = mock_post
        
        loader = MarkdownPromptLoader("test_prompts")
        prompt_file = Path("test.md")
        
        result = loader._parse_prompt_file(prompt_file)
        
        assert isinstance(result, PromptData)
        assert result.metadata.id == 1
        assert result.metadata.complexity == "low"
        assert result.content == "This is the prompt content"
        assert result.file_path == str(prompt_file)
    
    @patch('pathlib.Path.exists')
    def test_load_prompt_not_found(self, mock_exists):
        """Test loading non-existent prompt."""
        mock_exists.return_value = False
        
        loader = MarkdownPromptLoader("test_prompts")
        
        with pytest.raises(FileNotFoundError, match="Prompt 999 not found"):
            loader.load_prompt(999)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    @patch('frontmatter.load')
    def test_load_all_prompts(self, mock_frontmatter, mock_file, mock_glob, mock_exists):
        """Test loading all prompts from directory."""
        # Mock directory exists
        mock_exists.return_value = True
        
        # Mock glob returns two files
        mock_files = [Path("001.md"), Path("002.md")]
        mock_glob.return_value = mock_files
        
        # Mock frontmatter for two different prompts
        def frontmatter_side_effect(f):
            filename = f.name
            if "001.md" in filename:
                return type('MockPost', (), {
                    'metadata': {
                        'id': 1,
                        'complexity': 'low',
                        'target_mcp': ['test-mcp'],
                        'description': 'First prompt'
                    },
                    'content': 'First prompt content'
                })()
            else:  # 002.md
                return type('MockPost', (), {
                    'metadata': {
                        'id': 2,
                        'complexity': 'medium',
                        'target_mcp': ['test-mcp'],
                        'description': 'Second prompt'
                    },
                    'content': 'Second prompt content'
                })()
        
        mock_frontmatter.side_effect = frontmatter_side_effect
        
        loader = MarkdownPromptLoader("test_prompts")
        prompts = loader.load_all_prompts()
        
        assert len(prompts) == 2
        assert 1 in prompts
        assert 2 in prompts
        assert prompts[1].metadata.complexity == "low"
        assert prompts[2].metadata.complexity == "medium"
    
    def test_get_prompts_by_complexity(self):
        """Test filtering prompts by complexity."""
        loader = MarkdownPromptLoader("test_prompts")
        
        # Mock load_all_prompts
        mock_prompts = {
            1: PromptData(
                metadata=PromptMetadata(
                    id=1, complexity="low", 
                    target_mcp=["test"], description="Low complexity"
                ),
                content="Low content"
            ),
            2: PromptData(
                metadata=PromptMetadata(
                    id=2, complexity="medium", 
                    target_mcp=["test"], description="Medium complexity"
                ),
                content="Medium content"
            ),
            3: PromptData(
                metadata=PromptMetadata(
                    id=3, complexity="low", 
                    target_mcp=["test"], description="Another low"
                ),
                content="Another low content"
            )
        }
        
        with patch.object(loader, 'load_all_prompts', return_value=mock_prompts):
            low_prompts = loader.get_prompts_by_complexity("low")
            assert len(low_prompts) == 2
            assert all(p.metadata.complexity == "low" for p in low_prompts)
            
            medium_prompts = loader.get_prompts_by_complexity("medium")
            assert len(medium_prompts) == 1
            assert medium_prompts[0].metadata.complexity == "medium"
    
    def test_get_prompts_by_mcp(self):
        """Test filtering prompts by MCP target."""
        loader = MarkdownPromptLoader("test_prompts")
        
        mock_prompts = {
            1: PromptData(
                metadata=PromptMetadata(
                    id=1, complexity="low", 
                    target_mcp=["mcp-a", "mcp-b"], description="Multi MCP"
                ),
                content="Multi MCP content"
            ),
            2: PromptData(
                metadata=PromptMetadata(
                    id=2, complexity="medium", 
                    target_mcp=["mcp-b"], description="Single MCP"
                ),
                content="Single MCP content"
            )
        }
        
        with patch.object(loader, 'load_all_prompts', return_value=mock_prompts):
            mcp_b_prompts = loader.get_prompts_by_mcp("mcp-b")
            assert len(mcp_b_prompts) == 2
            
            mcp_a_prompts = loader.get_prompts_by_mcp("mcp-a")
            assert len(mcp_a_prompts) == 1
            
            mcp_c_prompts = loader.get_prompts_by_mcp("mcp-c")
            assert len(mcp_c_prompts) == 0
    
    def test_get_prompts_by_tags(self):
        """Test filtering prompts by tags."""
        loader = MarkdownPromptLoader("test_prompts")
        
        mock_prompts = {
            1: PromptData(
                metadata=PromptMetadata(
                    id=1, complexity="low", 
                    target_mcp=["test"], description="Tagged prompt",
                    tags=["basic", "test"]
                ),
                content="Tagged content"
            ),
            2: PromptData(
                metadata=PromptMetadata(
                    id=2, complexity="medium", 
                    target_mcp=["test"], description="Untagged prompt"
                ),
                content="Untagged content"
            ),
            3: PromptData(
                metadata=PromptMetadata(
                    id=3, complexity="high", 
                    target_mcp=["test"], description="Another tagged",
                    tags=["advanced", "test"]
                ),
                content="Advanced content"
            )
        }
        
        with patch.object(loader, 'load_all_prompts', return_value=mock_prompts):
            test_prompts = loader.get_prompts_by_tags(["test"])
            assert len(test_prompts) == 2
            
            basic_prompts = loader.get_prompts_by_tags(["basic"])
            assert len(basic_prompts) == 1
            
            multi_tag_prompts = loader.get_prompts_by_tags(["basic", "advanced"])
            assert len(multi_tag_prompts) == 2
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.write_text')
    def test_create_sample_prompt(self, mock_write, mock_mkdir):
        """Test creating sample prompt file."""
        loader = MarkdownPromptLoader("test_prompts")
        
        result = loader.create_sample_prompt(99, "output_dir")
        
        mock_mkdir.assert_called_once()
        mock_write.assert_called_once()
        
        written_content = mock_write.call_args[0][0]
        assert "id: 99" in written_content
        assert "Sample MCP Evaluation Prompt" in written_content
        assert result.endswith("099.md")
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_validate_prompt_directory(self, mock_glob, mock_exists):
        """Test prompt directory validation."""
        loader = MarkdownPromptLoader("test_prompts")
        
        # Mock directory exists
        mock_exists.return_value = True
        
        # Mock empty directory
        mock_glob.return_value = []
        
        report = loader.validate_prompt_directory()
        
        assert report["total_files"] == 0
        assert report["valid_prompts"] == 0
        assert report["invalid_prompts"] == 0
        assert report["errors"] == []
        assert report["duplicated_ids"] == []
    
    def test_invalid_complexity_validation(self):
        """Test validation rejects invalid complexity."""
        with pytest.raises(ValueError):
            loader = MarkdownPromptLoader("test_prompts")
            loader.get_prompts_by_complexity("invalid")


@pytest.fixture
def sample_prompt_content():
    """Sample prompt content for testing."""
    return """---
id: 1
complexity: "low"
target_mcp: ["test-mcp"]
description: "Sample test prompt"
expected_tools: ["Bash", "Read"]
tags: ["test", "sample"]
---

# Sample Test Prompt

This is a sample prompt for testing the loader functionality.

## Task
Test the prompt loader with various features.

## Success Criteria
- Metadata is properly parsed
- Content is extracted correctly
- All fields are accessible
"""


class TestIntegration:
    """Integration tests for prompt loader."""
    
    def test_full_prompt_workflow(self, tmp_path, sample_prompt_content):
        """Test complete workflow with real files."""
        # Create test prompts directory
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        
        # Create sample prompt file
        prompt_file = prompts_dir / "001.md"
        prompt_file.write_text(sample_prompt_content)
        
        # Initialize loader
        loader = MarkdownPromptLoader(str(prompts_dir))
        
        # Load prompt
        prompt = loader.load_prompt(1)
        
        # Verify metadata
        assert prompt.metadata.id == 1
        assert prompt.metadata.complexity == "low"
        assert prompt.metadata.target_mcp == ["test-mcp"]
        assert prompt.metadata.description == "Sample test prompt"
        assert prompt.metadata.timeout is None  # timeout is optional now
        assert prompt.metadata.expected_tools == ["Bash", "Read"]
        assert prompt.metadata.tags == ["test", "sample"]
        
        # Verify content
        assert "Sample Test Prompt" in prompt.content
        assert "Test the prompt loader" in prompt.content
        assert prompt.file_path == str(prompt_file)
        
        # Test caching
        prompt2 = loader.load_prompt(1)
        assert prompt is prompt2  # Same object from cache
