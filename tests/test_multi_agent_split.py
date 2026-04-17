"""Tests for multi_agent module split -- verify imports work from all paths."""


class TestImportsFromPackage:
    """Imports from multi_agent package directly."""

    def test_agent_definition(self):
        from multi_agent import AgentDefinition
        assert AgentDefinition is not None

    def test_subagent_task(self):
        from multi_agent import SubAgentTask
        assert SubAgentTask is not None

    def test_subagent_manager(self):
        from multi_agent import SubAgentManager
        assert SubAgentManager is not None

    def test_task_status(self):
        from multi_agent import TaskStatus
        assert TaskStatus is not None

    def test_load_agent_definitions(self):
        from multi_agent import load_agent_definitions
        assert callable(load_agent_definitions)

    def test_get_agent_definition(self):
        from multi_agent import get_agent_definition
        assert callable(get_agent_definition)


class TestImportsFromSubagentShim:
    """Backward-compatible imports from multi_agent.subagent."""

    def test_agent_definition(self):
        from multi_agent.subagent import AgentDefinition
        assert AgentDefinition is not None

    def test_subagent_manager(self):
        from multi_agent.subagent import SubAgentManager
        assert SubAgentManager is not None

    def test_task_status(self):
        from multi_agent.subagent import TaskStatus
        assert TaskStatus is not None


class TestImportsFromNewModules:
    """Direct imports from the new sub-modules."""

    def test_definitions(self):
        from multi_agent.definitions import AgentDefinition, get_agent_definition
        assert AgentDefinition is not None
        assert callable(get_agent_definition)

    def test_task(self):
        from multi_agent.task import SubAgentTask
        assert SubAgentTask is not None

    def test_manager(self):
        from multi_agent.manager import SubAgentManager
        assert SubAgentManager is not None


class TestConsistency:
    """Verify all import paths resolve to the same objects."""

    def test_same_agent_definition(self):
        from multi_agent import AgentDefinition as A1
        from multi_agent.subagent import AgentDefinition as A2
        from multi_agent.definitions import AgentDefinition as A3
        assert A1 is A3
        assert A2 is A3

    def test_same_manager(self):
        from multi_agent import SubAgentManager as M1
        from multi_agent.subagent import SubAgentManager as M2
        from multi_agent.manager import SubAgentManager as M3
        assert M1 is M3
        assert M2 is M3

    def test_same_task(self):
        from multi_agent import SubAgentTask as T1
        from multi_agent.subagent import SubAgentTask as T2
        from multi_agent.task import SubAgentTask as T3
        assert T1 is T3
        assert T2 is T3

    def test_same_task_status(self):
        from multi_agent import TaskStatus as S1
        from multi_agent.subagent import TaskStatus as S2
        from multi_agent.task import TaskStatus as S3
        assert S1 is S3
        assert S2 is S3


class TestBuiltinAgents:
    """Verify built-in agent definitions load correctly."""

    def test_load_returns_builtins(self):
        from multi_agent.definitions import load_agent_definitions
        agents = load_agent_definitions()
        assert "general-purpose" in agents
        assert "coder" in agents
        assert "reviewer" in agents
        assert "researcher" in agents
        assert "tester" in agents

    def test_get_known_agent(self):
        from multi_agent.definitions import get_agent_definition
        agent = get_agent_definition("coder")
        assert agent.name == "coder"
        assert len(agent.tools) > 0

    def test_get_unknown_agent_raises(self):
        import pytest
        from multi_agent.definitions import get_agent_definition
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent_definition("nonexistent-agent-type")
